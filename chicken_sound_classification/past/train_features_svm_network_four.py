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

# from aves_svm_network import AVESSVMNetwork
# from aves_svm_dataloader import AudioandLabelDataset
from model.features_svm_network import FeaturesSVMNetwork

SAMPLE_RATE = 16000

AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"


VAL_DATASET_CONFIG = "./dataset/embedding_label_dataset_20241121/val_dataset_config.csv"

TRAIN_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241203_four/full_dataset_config.csv"
# VAL_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241203_four/four_val_dataset_config.csv"

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    # L.seed_everything(42)

    # logger
    wandb_logger = WandbLogger(
        log_model=True,
    )

    all_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=10,
        filename="all_checkpoints/epoch={epoch:02d}-train_loss={train_loss:0.5f}-val_loss={val_loss:0.5f}",
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
        max_epochs=200,
        logger=wandb_logger,
        callbacks=callbacks,
        # profiler="simple",
    )

    estimator_network = FeaturesSVMNetwork(
        n_classes=4,
        train_dataset_config=TRAIN_DATASET_CONFIG,
        val_dataset_config=VAL_DATASET_CONFIG,
        test_dataset_config=VAL_DATASET_CONFIG,
        batch_size=4096,
        val_batch_size=1024,
        learning_rate=1e-3,
        embedding_dim=1024,
        audio_sr=16000,
    )
    
    # train
    trainer.fit(model=estimator_network, )
    del estimator_network.classifier_head
