import json
import torch
# import torch.nn as nn
import torch.utils
import torch.utils.data
from torchaudio.models import wav2vec2_model

import os
from typing import Tuple
from torch import optim, nn, utils, Tensor
from datetime import datetime
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
import lightning as L

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import matplotlib.pyplot as plt

from aves_svm_dataloader import AudioandLabelDataset

# define the LightningModule
class AVESSVMNetwork(L.LightningModule):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(
          self, 
          aves_config_path, 
          aves_model_path,
          aves_trainable,
          n_classes, 
          train_dataset_config,
          test_dataset_config,
          batch_size,
          learning_rate,
          embedding_dim=1024, 
          audio_sr = 16000,
        ):
        super().__init__()
        
        # training settings
        self.learning_rate = learning_rate

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.aves_config = self.load_aves_config(aves_config_path)
        self.aves_model = wav2vec2_model(**self.aves_config, aux_num_out=None)
        self.aves_model.load_state_dict(torch.load(aves_model_path))
        self.aves_model.to(self.device)
        # Freeze the AVES network
        self.aves_trainable = aves_trainable
        self.freeze_embedding_weights(self.aves_model, self.aves_trainable)
        self.aves_model.eval()
        # We will only train the classifier head
        self.classes_nums = n_classes
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes).to(self.device)
        # self.softmax = nn.Softmax()

        # audio settings
        self.sample_rate = audio_sr

        # dataset
        self.train_dataset_config = train_dataset_config
        self.test_dataset_config = test_dataset_config
        self.batch_size = batch_size

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

        # extract_features output: [features, None]
        # origin_features = self.aves_model.extract_features(x)
        # origin_features = self.aves_model.extract_features(x)[0]
        # print(f"self.aves_model.extract_features(x)[0]: {len(origin_features)}")
        # print(f"self.aves_model.extract_features(x)[0][0]: {self.aves_model.extract_features(x)[0][0].size()}")
        # print(f"self.aves_model.extract_features(x)[0][1]: {self.aves_model.extract_features(x)[0][1].size()}")
        
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        final_features = self.aves_model.extract_features(x)[0][-1]
        # print(f"dim features: {dim_features}")
        # print(f"len dim_features : {len(dim_features)}")
        # print(f"self.aves_model.extract_features(x)[0][-1] : {final_features.size()}")
        # print(f"origin_features : {origin_features}")
        # mean_embedding = final_features.mean(dim=1)
        # print(f"dim_features.mean(dim=1) : {mean_embedding.size()}")

        
        labels_hat = self.classifier_head(final_features)
        # soft_labels = self.softmax(labels)
        # print(f"len labels : {labels.size()}")

        return final_features, labels_hat

    def load_aves_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

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
      x, l = batch
      loss = 0

      features, l_hat = self(x)

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
          "x": x,
          "features": features,
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
        # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        class_names = ['Food Call', 'Pleasure Call', 'Distress Call', 'Other']
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
    
    # Code to use while initially setting up the model
    def freeze_embedding_weights(self, model, trainable):
      """ Freeze weights in AVES embeddings for classification """
      # The convolutional layers should never be trainable
      self.aves_model.feature_extractor.requires_grad_(False)
      self.aves_model.feature_extractor.eval()
      # The transformers are optionally trainable
      for param in self.aves_model.encoder.parameters():
        param.requires_grad = trainable
      if not trainable:
        # We also set layers without params (like dropout) to eval mode, so they do not change
        self.aves_model.encoder.eval()

    def train_dataloader(self):
      train_dataset = AudioandLabelDataset(
         config_path=self.train_dataset_config,
         partition="train",
        #  duration_sec=self.duration_sec,
         sample_rate=self.sample_rate,
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
      val_dataset = AudioandLabelDataset(
         config_path=self.test_dataset_config,
         partition="test",
        #  duration_sec=self.duration_sec,
         sample_rate=self.sample_rate,
        #  num_examples_per_epoch="",
      )

      g = torch.Generator(device="cuda")
      g.manual_seed(0)

      return torch.utils.data.DataLoader(
         val_dataset,
         num_workers=1,
         batch_size=106,
         generator=g,
         shuffle=False,
         persistent_workers=True,
      )
    
    # def test_dataloader(self):
    #   val_dataset = AudioandLabelDataset(
    #      config_path=self.test_dataset_config,
    #      partition="val",
    #     #  duration_sec=self.duration_sec,
    #      sample_rate=self.sample_rate,
    #     #  num_examples_per_epoch="",
    #   )

    #   g = torch.Generator(device="cuda")
    #   g.manual_seed(0)

    #   return torch.utils.data.DataLoader(
    #      val_dataset,
    #      num_workers=1,
    #      batch_size=self.batch_size,
    #      generator=g,
    #   )
