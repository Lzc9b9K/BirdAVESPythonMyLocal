# import json
import math
from argparse import ArgumentParser

import torch
import torch.utils
import torch.utils.data
# from torchaudio.models import wav2vec2_model

import os
from typing import Tuple
from torch import optim, nn
from datetime import datetime
import lightning as L

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import numpy as np
import matplotlib.pyplot as plt

# from aves_svm_dataloader import AudioandLabelDataset
from .features_svm_dataloader import EmbeddingandLabelDataset
from .classifier_model import FiveLayerNN, ThreeLayerNN, OneLayerNN

# スペクトルノルムを1にする初期化関数
def initialize_weights_with_spectral_norm(layer):
    if isinstance(layer, nn.Linear):
        # 標準的な初期化をまず適用
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        
        # スペクトルノルムを計算
        with torch.no_grad():
            u, s, v = torch.svd(layer.weight)
            spectral_norm = s.max()
            
            # スペクトルノルムで正規化
            layer.weight.data /= spectral_norm

class FeaturesSVMNetwork(L.LightningModule):
    """ Uses AVES embed features to classify """
    def __init__(
          self, 
          # aves_config_path, 
          # aves_model_path,
          # aves_trainable,
          n_classes, 
          train_dataset_config,
          val_dataset_config,
          test_dataset_config,
          batch_size,
          val_batch_size,
          learning_rate,
          embedding_dim=1024, 
          audio_sr = 16000,
          **kwargs
        ):
        super().__init__()
        
        # training settings
        self.learning_rate = learning_rate

        self.classes_nums = n_classes
        # self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes).to(self.device)
        # self.classifier_head = FiveLayerNN(input_size=embedding_dim, hidden_size=256, output_size=n_classes)
        self.classifier_head = ThreeLayerNN(input_size=embedding_dim, hidden_size=256, output_size=n_classes)
        # self.classifier_head = OneLayerNN(input_size=embedding_dim, output_size=n_classes)
        self.classifier_head.apply(initialize_weights_with_spectral_norm)
        # self.softmax = nn.Softmax()

        # audio settings
        self.sample_rate = audio_sr

        # dataset
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.test_dataset_config = test_dataset_config
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size

        # loss
        self.loss_function = torch.nn.CrossEntropyLoss()
        # val loss
        self.val_precision_list = []
        self.val_recall_list = []
        self.val_f_score_list = []
        self.val_attr_list = []

        # datetime
        self.train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # log hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        """
        Input
          x (Tensor): (batch, time)
        Returns
          mean_embedding (Tensor): (batch, output_dim)
          logits (Tensor): (batch, n_classes)
        """
        # print(f"x.size : {x.size()}")
        # batch_size = x.size(0)
        
        labels_hat = self.classifier_head(x)

        return labels_hat

    def common_paired_step(
        self,
        batch: Tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = False,
    ):
      """Model step used for validation and training.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing input audio (x) and target audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
                The firs optimizer corresponds to the generator and the second optimizer,
                corresponds to the adversarial loss (when in use).
            train (bool): Whether step is called during training (True) or validation (False).
      """
      e, l = batch
      loss = 0

      l_hat = self(e)

      loss = self.loss_function(l_hat, l)
      
      self.log(
        ("train" if train else "val") + "_loss",
        loss,
        prog_bar=True,
      )

      # store data dict
      data_dict = {
          "features": e.detach(),
          "l": l.detach(),
          "l_hat": l_hat.detach(),
      }

      return loss, data_dict

    def training_step(self, batch, batch_idx):
      loss, data_dict = self.common_paired_step(
        batch, 
        batch_idx,
        train=True)
      return loss

    def validation_step(self, batch, batch_idx):
      loss, data_dict = self.common_paired_step(
         batch, 
         batch_idx, 
         train=False)
      #TODO: ichishima data val anc cal
      attr, _, _ = batch
      data_dict["recording_number"] = attr["recording_number"]
      data_dict["age"] = attr["age"]

      event_windows = self.get_event_windows(data_dict["l"])
      predict_windows = self.get_event_windows(data_dict["l_hat"])
      matched_event_windows = self.find_matched_windows(data_dict["l"], data_dict["l_hat"])
      # Precssion, Recall, F-score
      precision = len(matched_event_windows) / len(predict_windows) if len(predict_windows) > 0 else 0
      recall = len(matched_event_windows) / len(event_windows) if len(event_windows) > 0 else 0
      f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

      self.val_precision_list.append(precision.detach().cpu())
      self.val_recall_list.append(recall.detach().cpu())
      self.val_f_score_list.append(f_score.detach().cpu())
      self.val_attr_list.append(data_dict)

      return data_dict

    def on_validation_end(self):
      try:
        self.val_precision_list = torch.concat(self.val_precision_list).numpy()
        self.val_recall_list = torch.concat(self.val_recall_list).numpy()
        self.val_f_score_list = torch.concat(self.val_f_score_list).numpy()
      except RuntimeError:
        self.val_precision_list = []
        self.val_recall_list = []
        self.val_f_score_list = []
        return super().on_validation_end()
      
      # Average & Mean Score
      self.log(
        "val_loss/Precision_mean",
        torch.mean(self.val_precision_list),
        prog_bar=True,
      )
      self.log(
        "val_loss/Recall_mean",
        torch.mean(self.val_recall_list),
        prog_bar=True,
      )
      self.log(
        "val_loss/F_score_mean",
        torch.mean(self.val_f_score_list),
        prog_bar=True,
      )
      # TODO: Average & Mean Score by Age Group
      age_groups = {0: [], 1: [], 2: [], 3: [], 4: []}
      for i, data_dict in enumerate(self.val_attr_list):
        age = int(data_dict["age"])
        if age in age_groups:
            age_groups[age].append((
               self.val_precision_list[i], 
               self.val_recall_list[i], 
               self.val_f_score_list[i]))
      for age, scores in age_groups.items():
        if scores:
            precisions, recalls, f_scores = zip(*scores)
            self.log(f"val_loss/Precision_age_{age}_mean", torch.mean(torch.tensor(precisions)), prog_bar=True)
            self.log(f"val_loss/Recall_age_{age}_mean", torch.mean(torch.tensor(recalls)), prog_bar=True)
            self.log(f"val_loss/F_score_age_{age}_mean", torch.mean(torch.tensor(f_scores)), prog_bar=True)

      # TODO: Average & Mean Score by Recording Number
      # TODO: Average & Mean Score by Recording Number & Age Group
      # TODO: Best Score Step Save
      
      # Clear lists for the next validation phase
      self.val_precision_list = []
      self.val_recall_list = []
      self.val_f_score_list = []
      return super().on_validation_end()

    def get_event_windows(self, labels):
      indices = torch.where(labels == 1)[0]
      event_windows = []
      current_window = []
      for idx in indices:
          if len(current_window) == 0:
              current_window = [idx, idx]
          elif idx == current_window[1] + 1:
              current_window[1] = idx
          else:
              event_windows.append(current_window)
              current_window = [idx, idx]
      if len(current_window) > 0 and current_window[0] != current_window[1]:
          event_windows.append(current_window)
      return event_windows

    def find_matched_windows(self, event_windows, predict_windows):
      matched_event_windows = []
      for event_start, event_end in event_windows:
          for predict_start, predict_end in predict_windows:
              if predict_start <= event_end and predict_end >= event_start:
                  matched_event_windows.append((event_start, event_end))
                  break
      return matched_event_windows

    def test_step(self, batch, batch_idx):
        loss, data_dict = self.common_paired_step(
           batch, 
           batch_idx, 
           train=False)
        
        labels_max_indices = torch.argmax(data_dict["l"], dim=-1)
        labels_hat_max_indices = torch.argmax(data_dict["l_hat"], dim=-1)
        # print(f"labels_max_indices: {labels_max_indices.size()}")
        # print(f"labels_hat_max_indices: {labels_hat_max_indices.size()}")
        correct_predictions = (labels_max_indices == labels_hat_max_indices)
        indices_accuracy = correct_predictions.float().mean().item()
        self.log("val_indices_acc", indices_accuracy, prog_bar=True)
        
        # calculate label accuracy
        correct_counts = torch.zeros(self.classes_nums, dtype=torch.int)
        total_counts = torch.zeros(self.classes_nums, dtype=torch.int)
        for i in range(self.classes_nums):
            total_counts[i] = (labels_max_indices == i).sum()
            correct_counts[i] = ((labels_max_indices == i) & (labels_hat_max_indices == i)).sum()
        accuracy = torch.where(total_counts > 0, correct_counts.float() / total_counts, torch.tensor(0.0))
        if (self.classes_nums == 4):
          class_names = ['Food Call', 'Pleasure Call', 'Distress Call', 'Other']
          self.log("val_label_foodCall_acc", accuracy[0], prog_bar=True)
          self.log("val_label_pleasureCall_acc", accuracy[1], prog_bar=True)
          self.log("val_label_distressCall_acc", accuracy[2], prog_bar=True)
          self.log("val_label_other_acc", accuracy[3], prog_bar=True)
        elif (self.classes_nums == 2):
          class_names = ['Food Call', 'Other']
          self.log("val_label_foodCall_acc", accuracy[0], prog_bar=True)
          self.log("val_label_other_acc", accuracy[1], prog_bar=True)
        else:
           pass

        # confusion matrix by quantities
        os.makedirs(f"./imgs/{self.train_datetime}", exist_ok=True)
        labels_np = labels_max_indices.view(-1).cpu().numpy()
        labels_hat_np = labels_hat_max_indices.view(-1).cpu().numpy()
        cm = confusion_matrix(labels_np, labels_hat_np)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f'Confusion Matrix with units as quantities of epoch {self.current_epoch}')
        plt.savefig(f'./imgs/{self.train_datetime}/confusion_matrix_epoch_{self.current_epoch}.png')
        plt.close()
        # confusion matrix by percentage
        cm = confusion_matrix(labels_np, labels_hat_np, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f'Confusion Matrix with units as percentage of epoch {self.current_epoch}')
        plt.savefig(f'./imgs/{self.train_datetime}/confusion_matrix_percentage_epoch_{self.current_epoch}.png')
        plt.close()

        return data_dict

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
      train_dataset = EmbeddingandLabelDataset(
         config_path=self.train_dataset_config,
         partition="train",
        #  duration_sec=self.duration_sec,
         duration_sample=1,
         duration_embedding=1,
        #  sample_rate=self.sample_rate,
        #  num_examples_per_epoch="",
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
         train_dataset,
         batch_size=self.batch_size,
         generator=g,
      )
    
    def val_dataloader(self):
      # val_dataset = ValidationEmbeddingandLabelDataset(
      #    config_path=self.val_dataset_config,
      #    partition="val",
      #   #  labels_num=self.classes_nums,
      #    classes_nums=self.classes_nums,
      # )
      val_dataset = EmbeddingandLabelDataset(
         config_path=self.test_dataset_config,
         partition="no",
        #  duration_sec=self.duration_sec,
         duration_sample=1,
         duration_embedding=1,
        #  sample_rate=self.sample_rate,
        #  num_examples_per_epoch="",
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
         val_dataset,
         num_workers=1,
         batch_size=self.val_batch_size,
        #  batch_size=4,
         generator=g,
         shuffle=False,
         persistent_workers=True,
      )
    
    def test_dataloader(self):
      test_dataset = EmbeddingandLabelDataset(
         config_path=self.train_dataset_config,
         partition="train",
        #  duration_sec=self.duration_sec,
         duration_sample=1,
         duration_embedding=1,
        #  sample_rate=self.sample_rate,
        #  num_examples_per_epoch="",
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
         test_dataset,
         batch_size=self.batch_size,
         generator=g,
      )
    
    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Logger ---
        parser.add_argument("--logger_project", type=str)
        parser.add_argument("--logger_name", type=str)
        parser.add_argument("--logger_save_dir", type=str)
        parser.add_argument("--cpkt_save_every_n_epochs", type=int, default=10)
        # --- Pytorch Lightning Trainer  ---
        parser.add_argument("--accelerator", type=str, default='gpu')
        parser.add_argument("--devices", type=int, default=1)
        # parser.add_argument("--gpus", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=400)  
        # parser.add_argument("--ckpt_path", type=str)
        # parser.add_argument("--resume_from_ckpt", type=int, default=0)
        # parser.add_argument("--log_folder_name", type=str)
        # --- Training  ---
        parser.add_argument("--n_classes", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=4096)
        parser.add_argument("--val_batch_size", type=int, default=4)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--audio_sr", type=int, default=16000)
        # parser.add_argument("--lr_scheduler", type=str, default="MultiStepLR")
        # parser.add_argument("--lr_patience", type=int, default=20)
        # --- Dataset ---
        parser.add_argument("--train_dataset_config", type=str)
        parser.add_argument("--val_dataset_config", type=str)
        parser.add_argument("--test_dataset_config", type=str)

        return parser
