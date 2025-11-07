# import os
import torch
# import pprint
# from argparse import ArgumentParser

# import torchaudio
# import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

# from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from model.features_svm_network import FeaturesSVMNetwork

SAMPLE_RATE = 16000

AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"

# TRAIN_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241202/train_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/processed_embedding_label_dataset_20241202/train_dataset_config.csv"
VAL_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241122/four_val_dataset_config.csv"

# MODEL_CKPT_PATH = "./model/pretrained_model/2024-12-02_model_4chuklpq/checkpoints/all_checkpoints/epoch=199-train_loss=0.00001-val_loss=0.00000.ckpt"
# MODEL_CKPT_PATH = "./model/pretrained_model/2024-12-03_model_8vwjnufk/checkpoints/all_checkpoints/epoch=199-train_loss=0.00082-val_loss=0.00000.ckpt"


# unprocessed 4
# TRAIN_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241203_four/full_dataset_config.csv"
# MODEL_CKPT_PATH = "./lightning_logs/qjjcd92s/checkpoints/all_checkpoints/epoch=199-train_loss=0.01014-val_loss=0.00000.ckpt"
# unprocessed 2
TRAIN_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241203_four/full_dataset_config.csv"
MODEL_CKPT_PATH = "./lightning_logs/qjjcd92s/checkpoints/all_checkpoints/epoch=199-train_loss=0.01014-val_loss=0.00000.ckpt"
# processed 2


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    # L.seed_everything(42)

    # logger
    wandb_logger = WandbLogger(
        log_model=True,
    )

    # all_checkpoint_callback = ModelCheckpoint(
    #     save_top_k=-1,
    #     every_n_epochs=10,
    #     filename="all_checkpoints/epoch={epoch:02d}-train_loss={train_loss:0.5f}-val_loss={val_loss:0.5f}",
    #     auto_insert_metric_name=False,
    # )

    # # trainer callbacks
    # callbacks = [
    #     all_checkpoint_callback
    # ]

    # create PyTorch Lightning trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1,
        logger=wandb_logger,
        # callbacks=callbacks,
        # profiler="simple",
    )

    estimator_network =  FeaturesSVMNetwork.load_from_checkpoint(
        MODEL_CKPT_PATH,
        val_dataset_config=VAL_DATASET_CONFIG,
        test_dataset_config=TRAIN_DATASET_CONFIG,
        batch_size=47999,
    )   

    trainer.test(estimator_network)
