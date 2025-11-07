import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import pprint
from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.new_features_svm_network import FeaturesSVMNetwork

import torch.multiprocessing as multiprocessing

SAMPLE_RATE = 16000

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("cuda is availabel")

    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
        print("{} setup done".format(multiprocessing.get_start_method()))

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    L.seed_everything(42)

    # some arg parse for configuration
    parser = ArgumentParser()
    parser = FeaturesSVMNetwork.add_model_specific_args(parser)
    args = parser.parse_args()
    pprint.pprint(f"args: {args}")

    # trainer callbacks
    wandb_logger = WandbLogger(
        project=args.logger_project,
        name=args.logger_name,
        save_dir=args.logger_save_dir,
        log_model=True,
    )
    all_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=args.cpkt_save_every_n_epochs,
        filename="all_checkpoints/epoch={epoch:02d}-train_loss={train_loss:0.5f}",
        auto_insert_metric_name=False,
    )
    callbacks = [
        all_checkpoint_callback
    ]

    # create PyTorch Lightning trainer          
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=args.cpkt_save_every_n_epochs,
        callbacks=callbacks,
        # profiler="simple",
    )

    estimator_network = FeaturesSVMNetwork(**vars(args))
    
    # train
    trainer.fit(model=estimator_network)
    del estimator_network.classifier_head
