import os
import torch
import pprint
from argparse import ArgumentParser

import torchaudio
# import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from aves_svm_network import AVESSVMNetwork
from aves_svm_dataloader import AudioandLabelDataset

SAMPLE_RATE = 16000

AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"

# TRAIN_DATASET_CONFIG = "./dataset/20mins_1s_compressed_dataset/train_dataset.csv"
# VAL_DATASET_CONFIG = "./dataset/20mins_1s_compressed_dataset/test_dataset.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20mins_2s_compressed_dataset/train_dataset.csv"
# VAL_DATASET_CONFIG = "./dataset/20mins_2s_compressed_dataset/test_dataset.csv"
TRAIN_DATASET_CONFIG = "./dataset/20mins_3s_compressed_dataset/train_dataset.csv"
VAL_DATASET_CONFIG = "./dataset/20mins_3s_compressed_dataset/test_dataset.csv"

if __name__ == "__main__":
    # run(
    #   # dataset_dataframe=df,
    #   train_dataset_config="./dataset/train/0_train_dataset.csv",
    #   test_dataset_config="./dataset/test/0_test_dataset.csv",
    #   model_path="./content/birdaves-biox-large.torchaudio.pt",
    #   model_config_path="./content/birdaves-biox-large.torchaudio.model_config.json",
    #   duration_sec=1.0,
    #   # annotation_name="call_type",
    #   learning_rate=1e-3,
    #   batch_size=8,
    #   n_epochs=10,
    #   n_classes=4
    # )

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    L.seed_everything(42)

    # logger
    # logger = TensorBoardLogger(
    #     save_dir="./logs/1108",
    #     name="01",
    #     # default_hp_metric=False,
    # )
    wandb_logger = WandbLogger(
        log_model=True,
    )

    # monitor and logs learning rate
    # lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # trainer callbacks
    # callbacks = [
    #     lr_monitor,
    # ]

    # create PyTorch Lightning trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=10,
        logger=wandb_logger,
        # callbacks=callbacks,
        # profiler="simple",
    )

    estimator_network = AVESSVMNetwork(
        aves_config_path=AVES_MODEL_CONFIG_PATH, 
        aves_model_path=AVES_MODEL_PATH,
        aves_trainable=False,
        n_classes=4,
        train_dataset_config=TRAIN_DATASET_CONFIG,
        test_dataset_config=VAL_DATASET_CONFIG,
        batch_size=32,
        learning_rate=1e-3,
        embedding_dim=1024,
        audio_sr=16000,
    )

    # # test
    # audio_1s_data = "./dataset/20mins_1s_compressed_dataset/full/00_00_15_312.wav" 
    # x, sr = torchaudio.load(audio_1s_data)
    # output = estimator_network(x)

    # dataset
    # g = torch.Generator(device="cuda")
    # g.manual_seed(0)
    
    # train_dataset = AudioandLabelDataset(
    #      config_path="./dataset/20mins_1s_compressed_dataset/train_dataset.csv",
    #      partition="train",
    #     #  duration_sec=self.duration_sec,
    #      sample_rate=SAMPLE_RATE,
    #     #  num_examples_per_epoch="",
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     # batch_size=self.batch_size,
    #     batch_size=32,
    #     generator=g,
    # )
    # val_dataset = AudioandLabelDataset(
    #      config_path="./dataset/20mins_1s_compressed_dataset/test_dataset.csv",
    #      partition="test",
    #     #  duration_sec=self.duration_sec,
    #      sample_rate=SAMPLE_RATE,
    #     #  num_examples_per_epoch="",
    # )
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=32,
    #     generator=g,
    # )
    
    # train
    trainer.fit(
        model=estimator_network, 
        # train_dataloaders=train_dataloader, 
        # val_dataloaders=val_dataloader,
        )
    del estimator_network.classifier_head
