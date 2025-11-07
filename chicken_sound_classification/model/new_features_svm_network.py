# import json
import math
from argparse import ArgumentParser
from collections import defaultdict

import torch
import torch.utils
import torch.utils.data
# from torchaudio.models import wav2vec2_model

import os
from typing import Tuple
from torch import optim, nn
from datetime import datetime
import lightning as L
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from .new_features_svm_dataloader import EmbeddingandLabelDataset, ValidationEmbeddingandLabelDataset
from .classifier_model import FiveLayerNN, ThreeLayerNN, OneLayerNN
from .utils import seed_worker

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
          **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()

        # self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes).to(self.device)
        # self.classifier_head = FiveLayerNN(input_size=self.hparams.embedding_dim, hidden_size=256, output_size=self.hparams.classes_nums)
        self.classifier_head = ThreeLayerNN(input_size=self.hparams.embedding_dim, hidden_size=256, output_size=self.hparams.classes_nums)
        # self.classifier_head = OneLayerNN(input_size=embedding_dim, output_size=n_classes)
        self.classifier_head.apply(initialize_weights_with_spectral_norm)
        # self.softmax = nn.Softmax()

        # loss
        self.loss_function = torch.nn.CrossEntropyLoss()
        # val loss
        # self.val_accuracy_list = []
        # self.val_precision_list = []
        # self.val_recall_list = []
        # self.val_f_score_list = []
        self.val_accuracy_metric = BinaryAccuracy(device="cuda")
        self.val_precision_metric = BinaryPrecision(device="cuda")
        self.val_recall_metric = BinaryRecall(device="cuda")
        self.val_f_score_metric = BinaryF1Score(device="cuda")
        self.val_attr_list = []

        # datetime
        self.train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
      _, e, l = batch
      loss = 0

      print(f"embedding_tensor: {e} \n")
      print(f"label_tensor: {l} \n")

      l_hat = self(e)
      # if ( train == True) :
      #   print("\n train step l_hat.shape", {l_hat.shape})
      #   print("\n train step l.shape", {l.shape})
      # else:
      #   print("\n val step l_hat.shape", {l_hat.shape})
      #   print("\n val step l.shape", {l.shape})

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
      # data_dict["recording_number"] = attr["recording_number"]
      # data_dict["age"] = attr["age"]

      # one-hot to scalar
      # print("\n data_dict l", data_dict["l"].shape)
      # print("\n data_dict l_hat", data_dict["l_hat"].shape)
      l = torch.argmax(data_dict["l"], dim=2)      
      l_hat = torch.argmax(data_dict["l_hat"], dim=2)
      # print("l.shape", l.shape)
      # print("l_hat.shape", l_hat.shape)

      # update mertirc
      # print("l.device", l.device)
      # print("l_hat.device", l_hat.device)
      l_view = l.view(-1).detach()
      l_hat_view = l_hat.view(-1).detach()
      # print("l_view.device", l_view.device)
      # print("l_hat_view.device", l_hat_view.device)

      self.val_accuracy_metric.update(l_hat_view, l_view)
      self.val_precision_metric.update(l_hat_view, l_view)
      self.val_recall_metric.update(l_hat_view, l_view)
      self.val_f_score_metric.update(l_hat_view, l_view)

      # attr_item = {
      #     "recording_number": attr["recording_number"],
      #     "age": attr["age"],
      #     "l": l,
      #     "l_hat": l_hat  
      # }
      # self.val_attr_list.append(attr_item)
      for i in range(l.size(0)):  # 遍历 batch 中的每个样本
        # print("print(attr['recording_number'][i])", attr["recording_number"][i])
        # print("print(attr['age'][i])", attr["age"][i])
        attr_item = {
            "recording_number": attr["recording_number"][i],
            "age": attr["age"][i],
            "l": l[i].detach(),
            "l_hat": l_hat[i].detach(),
        }
        self.val_attr_list.append(attr_item)
        # print("\n attr_item", attr_item)

      return super().validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
      # Average & Mean Score
      accuracy = self.val_accuracy_metric.compute()
      precision = self.val_precision_metric.compute()
      recall = self.val_recall_metric.compute()
      f1_score = self.val_f_score_metric.compute()
      self.log("val_loss/Accuracy", accuracy, prog_bar=True)
      self.log("val_loss/Precision", precision, prog_bar=True)
      self.log("val_loss/Recall", recall, prog_bar=True)
      self.log("val_loss/F_score", f1_score, prog_bar=True)
      
      # TODO: Average & Mean Score by Age Group
      grouped_results = defaultdict(lambda: {
          "accuracy": BinaryAccuracy(),
          "precision": BinaryPrecision(),
          "recall": BinaryRecall(),
          "f1": BinaryF1Score()
      })
      age_groups = {0: [], 1: [], 2: [], 3: [], 4: []}
      # 根据 age 进行分组
      # print("\n self.val_attr_list", self.val_attr_list)
      for attr_item in self.val_attr_list:
          l_hat = attr_item["l_hat"]
          l = attr_item["l"]
          age = attr_item["age"]
          for group in age_groups:
              if age == group:
                  grouped_results[group]["accuracy"].update(l_hat, l)
                  grouped_results[group]["precision"].update(l_hat, l)
                  grouped_results[group]["recall"].update(l_hat, l)
                  grouped_results[group]["f1"].update(l_hat, l)
      # 计算每个组的指标
      for group, metrics in grouped_results.items():
          group_accuracy = metrics["accuracy"].compute()
          group_precision = metrics["precision"].compute()
          group_recall = metrics["recall"].compute()
          group_f1 = metrics["f1"].compute()
          # print(f"Group {group} - Accuracy: {group_accuracy}, Precision: {group_precision}, Recall: {group_recall}, F1 Score: {group_f1}")                
          self.log(f"val_loss/Accuracy_age_{group}", torch.mean(torch.tensor(group_accuracy)), prog_bar=True)
          self.log(f"val_loss/Precision_age_{group}", torch.mean(torch.tensor(group_precision)), prog_bar=True)
          self.log(f"val_loss/Recall_age_{group}", torch.mean(torch.tensor(group_recall)), prog_bar=True)
          self.log(f"val_loss/F_score_age_{group}", torch.mean(torch.tensor(group_f1)), prog_bar=True)

      # TODO: Average & Mean Score by Recording Number
      # TODO: Average & Mean Score by Recording Number & Age Group
      # TODO: Best Score Step Save
      
      # Clear lists for the next validation phase]
      self.val_accuracy_metric.reset()
      self.val_precision_metric.reset()
      self.val_recall_metric.reset()
      self.val_f_score_metric.reset()
      self.val_attr_list = []
      return super().on_validation_end()

    # def test_step(self, batch, batch_idx):
    #   loss, data_dict = self.common_paired_step(
    #      batch, 
    #      batch_idx, 
    #      train=False)
    #   #TODO: ichishima data val anc cal
    #   attr, _, _ = batch
    #   data_dict["recording_number"] = attr["recording_number"]
    #   data_dict["age"] = attr["age"]

    #   event_windows = self.get_event_windows(data_dict["l"])
    #   # print("\n event_windows.shape", event_windows.shape, flush=True)
    #   print("\n event_windows", event_windows, flush=True)
    #   predict_windows = self.get_event_windows(data_dict["l_hat"])
    #   # print("\n predict_windows.shape", predict_windows.shape, flush=True)
    #   matched_event_windows = self.find_matched_windows(data_dict["l"], data_dict["l_hat"])
    #   # Precssion, Recall, F-score
    #   precision = len(matched_event_windows) / len(predict_windows) if len(predict_windows) > 0 else 0
    #   recall = len(matched_event_windows) / len(event_windows) if len(event_windows) > 0 else 0
    #   f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    #   self.val_precision_list.append(precision.detach().cpu())
    #   self.val_recall_list.append(recall.detach().cpu())
    #   self.val_f_score_list.append(f_score.detach().cpu())
    #   self.val_attr_list.append(data_dict)

    #   return data_dict

    # def on_test_end(self):
    #   try:
    #     self.val_precision_list = torch.concat(self.val_precision_list).numpy()
    #     self.val_recall_list = torch.concat(self.val_recall_list).numpy()
    #     self.val_f_score_list = torch.concat(self.val_f_score_list).numpy()
    #   except RuntimeError:
    #     self.val_precision_list = []
    #     self.val_recall_list = []
    #     self.val_f_score_list = []
    #     return super().on_validation_end()
      
    #   # Average & Mean Score
    #   self.log(
    #     "val_loss/Precision_mean",
    #     torch.mean(self.val_precision_list),
    #     prog_bar=True,
    #   )
    #   self.log(
    #     "val_loss/Recall_mean",
    #     torch.mean(self.val_recall_list),
    #     prog_bar=True,
    #   )
    #   self.log(
    #     "val_loss/F_score_mean",
    #     torch.mean(self.val_f_score_list),
    #     prog_bar=True,
    #   )
    #   # TODO: Average & Mean Score by Age Group
    #   age_groups = {0: [], 1: [], 2: [], 3: [], 4: []}
    #   for i, data_dict in enumerate(self.val_attr_list):
    #     age = int(data_dict["age"])
    #     if age in age_groups:
    #         age_groups[age].append((
    #            self.val_precision_list[i], 
    #            self.val_recall_list[i], 
    #            self.val_f_score_list[i]))
    #   for age, scores in age_groups.items():
    #     if scores:
    #         precisions, recalls, f_scores = zip(*scores)
    #         self.log(f"val_loss/Precision_age_{age}_mean", torch.mean(torch.tensor(precisions)), prog_bar=True)
    #         self.log(f"val_loss/Recall_age_{age}_mean", torch.mean(torch.tensor(recalls)), prog_bar=True)
    #         self.log(f"val_loss/F_score_age_{age}_mean", torch.mean(torch.tensor(f_scores)), prog_bar=True)

    #   # TODO: Average & Mean Score by Recording Number
    #   # TODO: Average & Mean Score by Recording Number & Age Group
    #   # TODO: Best Score Step Save
      
    #   # Clear lists for the next validation phase
    #   self.val_precision_list = []
    #   self.val_recall_list = []
    #   self.val_f_score_list = []
    #   return super().on_validation_end()

    # def get_event_windows(self, labels):
    #   indices = torch.where(labels == 1)[0]
    #   event_windows = []
    #   current_window = []
    #   for idx in indices:
    #       if len(current_window) == 0:
    #           current_window = [idx, idx]
    #       elif idx == current_window[1] + 1:
    #           current_window[1] = idx
    #       else:
    #           event_windows.append(current_window)
    #           current_window = [idx, idx]
    #   if len(current_window) > 0 and current_window[0] != current_window[1]:
    #       event_windows.append(current_window)
    #   if event_windows:
    #       event_windows_tensor = torch.tensor(event_windows, dtype=torch.long)
    #   else:
    #       event_windows_tensor = torch.empty((0, 2), dtype=torch.long)
    #   return event_windows_tensor

    # def find_matched_windows(self, event_windows, predict_windows):
    #   matched_event_windows = []
    #   for event_window in event_windows[0]:
    #     # print("\n event_window", event_windows)
    #     event_start, event_end = event_window.tolist()
    #     for predict_window in predict_windows[0]:
    #       predict_start, predict_end = predict_window.tolist() 
    #       if (predict_start <= event_end) and (predict_end >= event_start):
    #           matched_event_windows.append((event_start, event_end))
    #           break
    #   # Convert the list of matched windows back to a tensor
    #   if matched_event_windows:
    #       matched_event_windows_tensor = torch.stack(matched_event_windows)
    #   else:
    #       matched_event_windows_tensor = torch.empty((0, 2), dtype=torch.long)
    #   return matched_event_windows_tensor

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def train_dataloader(self):
      train_dataset = EmbeddingandLabelDataset(
         config_path=self.hparams.train_dataset_config,
         partition="train",
         duration_sample=1,
         duration_embedding=1,
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.hparams.train_batch_size,
        worker_init_fn=seed_worker,
        num_workers=self.hparams.train_num_workers,
        generator=g,
        persistent_workers=True,
        # pin_memory=True,
      )
    
    def val_dataloader(self):
      val_dataset = ValidationEmbeddingandLabelDataset(
         config_path=self.hparams.val_dataset_config,
         partition="val",
         classes_nums=self.hparams.classes_nums,
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
         val_dataset,
         batch_size=self.hparams.val_batch_size,
        #  batch_size=4,
         generator=g,
         num_workers=self.hparams.val_num_workers,
         shuffle=False,
         persistent_workers=True,
      )
    
    # def test_dataloader(self):
    #   test_dataset = EmbeddingandLabelDataset(
    #      config_path=self.hparams.train_dataset_config,
    #      partition="train",
    #      duration_sample=1,
    #      duration_embedding=1,
    #   )

    #   g = torch.Generator(device="cuda")
    #   g.manual_seed(0)

    #   return torch.utils.data.DataLoader(
    #      test_dataset,
    #      batch_size=self.hparams.batch_size,
    #      generator=g,
    #   )
    
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
        parser.add_argument("--classes_nums", type=int, default=2)
        parser.add_argument("--train_batch_size", type=int, default=4096)
        parser.add_argument("--val_batch_size", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--embedding_dim", type=int, default=1024)
        parser.add_argument("--sample_rate", type=int, default=16000)
        parser.add_argument("--train_num_workers", type=int, default=23)
        parser.add_argument("--val_num_workers", type=int, default=23)
        # parser.add_argument("--lr_scheduler", type=str, default="MultiStepLR")
        # parser.add_argument("--lr_patience", type=int, default=20)
        # --- Dataset ---
        parser.add_argument("--train_dataset_config", type=str)
        parser.add_argument("--val_dataset_config", type=str)
        parser.add_argument("--test_dataset_config", type=str)

        return parser
