# import os
import torch
# import pprint
# from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint


from model.features_svm_network import FeaturesSVMNetwork

SAMPLE_RATE = 16000

AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"

# VAL_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241122/val_dataset_config.csv"
VAL_DATASET_CONFIG = "./dataset/val_data/original_val_dataset_config.csv"

# 2024/12/06
# TRAIN_DATASET_CONFIG = "./dataset/20241205_original_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241205_highPassed_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241205_bandPassed_dataset/full_dataset_config.csv"
# 2024/12/18
# TRAIN_DATASET_CONFIG = "./dataset/20241218_original_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241218_original_dataset_4/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241218_highPassed_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241218_highPassed_dataset_4/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241218_bandPassed_dataset/full_dataset_config.csv"
# 2024/12/25
# TRAIN_DATASET_CONFIG = "./dataset/20241219_original_dataset_2/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241219_highPassed_dataset_2/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241205_bandPassed_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241205_original_dataset/full_dataset_config.csv"
# TRAIN_DATASET_CONFIG = "./dataset/20241205_highPassed_dataset/full_dataset_config.csv"

# stft
# TRAIN_DATASET_CONFIG = "./dataset/2025_stft_dataset/original/full_dataset_config_o_2.csv"
# TRAIN_DATASET_CONFIG = "./dataset/2025_stft_dataset/bandPassed/full_dataset_config_bp_2.csv"

# 2025/06/10
TRAIN_DATASET_CONFIG = "./dataset/train_data/bandPassedDatas/bandPassed_full_dataset_config.csv"


if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    # L.seed_everything(42)

    # logger
    wandb_logger = WandbLogger(
        project="stft_3layer_model",
        # name="original_dataset_2",
        # name="highPassed_daset_2",
        name="bandPassed_dataset_2",
        # name="original_dataset_4",
        # name="highPassed_dataset_4",
        log_model=True,
    )

    all_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=10,
        filename="all_checkpoints/epoch={epoch:02d}-train_loss={train_loss:0.5f}",
        auto_insert_metric_name=False,
    )

    # trainer callbacks
    callbacks = [
        all_checkpoint_callback
    ]

    # create PyTorch Lightning trainer          
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        # max_epochs=200,
        # max_epochs=600,
        max_epochs=100,
        logger=wandb_logger,
        callbacks=callbacks,
        # profiler="simple",
    )

    estimator_network = FeaturesSVMNetwork(
        n_classes=2,
        # n_classes=4,
        train_dataset_config=TRAIN_DATASET_CONFIG,
        val_dataset_config=VAL_DATASET_CONFIG,
        test_dataset_config=VAL_DATASET_CONFIG,
        batch_size=4096,
        val_batch_size=4,
        learning_rate=1e-3,
        embedding_dim=1024,
        # embedding_dim=513,
        audio_sr=16000,
    )
    
    # train
    trainer.fit(model=estimator_network)
    del estimator_network.classifier_head
