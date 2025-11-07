import json
import math

import torch
import torch.utils
import torch.utils.data
from torchaudio.models import wav2vec2_model

import os
from typing import Tuple
from torch import optim, nn
from datetime import datetime
import lightning as L

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
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

# define the LightningModule
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
        ):
        super().__init__()
        
        # training settings
        self.learning_rate = learning_rate

        self.classes_nums = n_classes
        # self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes).to(self.device)
        # self.classifier_head = FiveLayerNN(input_size=embedding_dim, hidden_size=256, output_size=n_classes)
        # self.classifier_head = ThreeLayerNN(input_size=embedding_dim, hidden_size=256, output_size=n_classes)
        self.classifier_head = OneLayerNN(input_size=embedding_dim, hidden_size=256, output_size=n_classes)
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

      # print(f"l : {l.size()}")
      # print(f"l_hat : {l_hat.size()}")
      

      loss = self.loss_function(l_hat, l)
      # print(f"loss : {loss}")
      
      self.log(
        ("train" if train else "val") + "_loss",
        loss,
        prog_bar=True,
      )
      # self.log("train_loss", loss)

      # store data dict
      data_dict = {
          "features": e,
          "l": l,
          "l_hat": l_hat,
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

        # calculate acc
        # labels = torch.argmax(data_dict["l"], dim=-1)
        # labels_hat = torch.argmax(data_dict["l_hat"], dim=-1)
        # print(f"labels: {labels.size()}")
        # print(f"labels_hat: {labels_hat.size()}")
        # val_acc = torch.sum(labels == labels_hat).item() / (len(labels) * 1.0)
        # data_dict["val_acc"] = val_acc
        # # print(f"val_acc : val_acc")
        # self.log("val_acc", val_acc, prog_bar=True)
        # calculate indices accuracy
        labels_max_indices = torch.argmax(data_dict["l"], dim=-1)
        labels_hat_max_indices = torch.argmax(data_dict["l_hat"], dim=-1)
        # print(f"labels_max_indices: {labels_max_indices.size()}")
        # print(f"labels_hat_max_indices: {labels_hat_max_indices.size()}")
        correct_predictions = (labels_max_indices == labels_hat_max_indices)
        indices_accuracy = correct_predictions.float().mean().item()
        self.log("val_indices_acc", indices_accuracy, prog_bar=True)
        
        # calculate label accuracy
        correct_counts = torch.zeros(4, dtype=torch.int)
        total_counts = torch.zeros(4, dtype=torch.int)
        for i in range(4):
          total_counts[i] = (labels_max_indices == i).sum()
          correct_counts[i] = ((labels_max_indices == i) & (labels_hat_max_indices == i)).sum()
        accuracy = torch.where(total_counts > 0, correct_counts.float() / total_counts, torch.tensor(0.0))
        self.log("val_label_foodCall_acc", accuracy[0], prog_bar=True)
        self.log("val_label_pleasureCall_acc", accuracy[1], prog_bar=True)
        self.log("val_label_distressCall_acc", accuracy[2], prog_bar=True)
        self.log("val_label_other_acc", accuracy[3], prog_bar=True)

        # confusion matrix by quantities
        os.makedirs(f"./imgs/{self.train_datetime}", exist_ok=True)
        labels_np = labels_max_indices.view(-1).cpu().numpy()
        labels_hat_np = labels_hat_max_indices.view(-1).cpu().numpy()
        cm = confusion_matrix(labels_np, labels_hat_np)
        class_names = ['Food Call', 'Pleasure Call', 'Distress Call', 'Other']
        # class_names = ['Food Call', 'Other']
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
    
