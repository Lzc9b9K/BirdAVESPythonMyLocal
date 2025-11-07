from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import DataLoader

from new_chicken_sound_dataloader import Vox
from new_chicken_sound_model import AvesClassifier, set_train_aves, set_eval_aves

def train_one_epoch(model, dataloader, optimizer, loss_function):
    """ Update model based on supervised classification task """

    set_train_aves(model)
    loss_function = nn.CrossEntropyLoss()

    epoch_losses = []
    iterator = tqdm(dataloader)
    for i, batch_dict in enumerate(iterator):
        optimizer.zero_grad()
        if torch.cuda.is_available():
          batch_dict["x"] = batch_dict["x"].cuda()
          batch_dict[dataloader.dataset.annotation_name] = batch_dict[dataloader.dataset.annotation_name].cuda()

        embedding, logits = model(batch_dict["x"])
        # print("logits", len(logits))
        # print("batch_dict[dataloader.dataset.annotation_name]", batch_dict[dataloader.dataset.annotation_name])
        loss = loss_function(logits, batch_dict[dataloader.dataset.annotation_name].to(torch.long))

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        if len(epoch_losses) > 10:
          iterator.set_description(f"Train loss: {np.mean(epoch_losses[-10:]):.3f}")

    return epoch_losses

def test_one_epoch(model, dataloader, loss_function, epoch_idx):
  """ Obtain loss and F1 scores on test set """

  set_eval_aves(model)

  # Obtain predictions
  all_losses = []
  all_predictions = []
  with torch.no_grad():
    for i, batch_dict in enumerate(dataloader):
        if torch.cuda.is_available():
          batch_dict["x"] = batch_dict["x"].cuda()
          batch_dict[dataloader.dataset.annotation_name] = batch_dict[dataloader.dataset.annotation_name].cuda()
        embedding, logits = model(batch_dict["x"])
        all_losses.append(loss_function(logits, batch_dict[dataloader.dataset.annotation_name].to(torch.long)))
        all_predictions.append(logits.argmax(1))
        # print(dataloader)
        # print("i", i)

  # Format predictions and annotations
  all_losses = torch.stack(all_losses)
  all_predictions = torch.cat(all_predictions).cpu().numpy()
  all_annotations = dataloader.dataset.dataset_info[dataloader.dataset.annotation_name + "_int"].to_numpy() # since dataloader shuffle = False
  # Get confusion matrix
  # print("all_annotations", all_annotations)
  # print("len(all_predictions)", len(all_predictions))

  # cm = confusion_matrix(all_annotations, all_predictions)
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataloader.dataset.classes)
  # disp.plot()
  # disp.ax_.set_title(f"Test epoch {epoch_idx}")
  
  # Compute Accuracy
  correct_predictions = np.sum(all_annotations == all_predictions)
  total_predictions = all_annotations.shape[0]
  accuracy = correct_predictions / total_predictions
  print(f"Accuracy: {accuracy:.2f}")
  # Compute F1
  f1_scores = f1_score(all_annotations, all_predictions, average=None)
  macro_average_f1 = f1_score(all_annotations, all_predictions, average="macro")
  # Report
  print(f"Mean test loss: {all_losses.mean():.3f}, Macro-average F1: {macro_average_f1:.3f}")
  print("F1 by class:")
  print({k: np.round(s,decimals=4) for (k,s) in zip(dataloader.dataset.classes, f1_scores)})
  
  cm = confusion_matrix(all_annotations, all_predictions)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataloader.dataset.classes)
  disp.plot()
  disp.ax_.set_title(f"Test epoch {epoch_idx}")
  
  # return

def run(
      # dataset_dataframe,
      train_dataset_config,
      test_dataset_config,
      model_path,
      model_config_path,
      duration_sec,
      annotation_name,
      learning_rate,
      batch_size,
      n_epochs,
      aves_sr = 16000
      ):

  print("Setting up dataloaders")
  # train_dataloader = get_dataloader(dataset_dataframe, True, aves_sr, duration_sec, annotation_name, batch_size)
  # test_dataloader = get_dataloader(dataset_dataframe, False, aves_sr, duration_sec, annotation_name, batch_size)
  train_dataloader = DataLoader(
     Vox(config_path=train_dataset_config, partition="train", duration_sec=duration_sec, sample_rate=aves_sr, num_examples_per_epoch=batch_size),
     batch_size=batch_size,
     shuffle=True,
     drop_last=True,
  )
  test_dataloader = DataLoader(
     Vox(config_path=test_dataset_config, partition="test", duration_sec=duration_sec, sample_rate=aves_sr, num_examples_per_epoch=batch_size),
     batch_size=batch_size,
     shuffle=False,
     drop_last=False,
  )

  print("Setting up model")
  model = AvesClassifier(
     config_path=model_config_path, 
     model_path=model_path, 
     n_classes=len(train_dataloader.dataset.classes), 
     trainable=False,
     embedding_dim=1024,
     audio_sr = 16000,
     )
  if torch.cuda.is_available():
    model.cuda()

  print("Setting up optimizers")
  optimizer = torch.optim.Adam(model.classifier_head.parameters(), lr=learning_rate)

  print("Setting up loss function")
  loss_function = nn.CrossEntropyLoss()

  for epoch_idx in range(n_epochs):
    print(f"~~ Training epoch {epoch_idx}")
    train_one_epoch(model, train_dataloader, optimizer, loss_function)
    print(f"~~ Testing epoch {epoch_idx}")
    test_one_epoch(model, test_dataloader, loss_function, epoch_idx)

  train_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  torch.save(model.state_dict(), f"./save_pts/{train_datetime}_class_ckpt.pt")

  return

if __name__ == "__main__":
   run(
      # dataset_dataframe=df,
      train_dataset_config="./dataset/train/0_train_dataset.csv",
      test_dataset_config="./dataset/test/0_test_dataset.csv",
      model_path="./content/birdaves-biox-large.torchaudio.pt",
      model_config_path="./content/birdaves-biox-large.torchaudio.model_config.json",
      duration_sec=1.0,
      annotation_name="call_type",
      learning_rate=1e-3,
      batch_size=16,
      n_epochs=100
    )
