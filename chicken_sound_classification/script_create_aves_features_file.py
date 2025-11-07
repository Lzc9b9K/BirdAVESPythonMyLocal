import os
import shutil
import scipy.io.wavfile as wav
import random
from datetime import datetime

import h5py
import numpy as np

import torch
import torchaudio

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from aves_svm_network import AVESSVMNetwork

SAMPLE_RATE = 16000

AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"

DATA_DIR = "./dataset"

TRAIN_DATASET_CONFIG = "./dataset/20mins_3s_compressed_dataset/train_dataset.csv"
VAL_DATASET_CONFIG = "./dataset/20mins_3s_compressed_dataset/test_dataset.csv"

def save_as_hdf5(tensor_data, output_file):
    file_path = output_file
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset("tensor_dataset", data=tensor_data.cpu().numpy())
    print(f"Save as {file_path}")

def replace_wav_with_h5(filename):
    base_name, ext = os.path.splitext(filename)
    if ext.lower() == ".wav":
        new_filename = f"{base_name}.h5"
        return new_filename
    else:
        return None
    
def process_wav_to_h5_in_directory(estimator_network, root_directory, output_root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                # torch load
                # print(f"dirpath: {dirpath}")
                # print(f"dirnames: {dirnames}")
                # print(f"filename: {filename}")
                audio_file_path = os.path.join(dirpath, filename)
                x, sr = torchaudio.load(audio_file_path)

                # model processing
                output_features, _ = estimator_network(x)
                output_features_squeezed = output_features.squeeze()

                # save hdf5
                hdf5_file_name = replace_wav_with_h5(filename)
                relative_path = os.path.relpath(dirpath, root_directory)
                # print(f"relative_path : {relative_path}")
                # hdf5_file_save_path = f"{output_root_directory}/{relative_path}/{hdf5_file_name}"
                target_dir = os.path.join(output_root_directory, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                hdf5_file_save_path = os.path.join(target_dir, hdf5_file_name)
                # save
                save_as_hdf5(output_features_squeezed, hdf5_file_save_path)

if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    L.seed_everything(42)

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
    estimator_network.eval()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"script begin: {start_time_str}")

    # process
    # root_dir = f"{DATA_DIR}/ichishima_datas/hen_1_chick_1_5th_time"
    # output_dir = f"{DATA_DIR}/ichishima_datas/BirdAVES_features/hen_1_chick_1_5th_time"
    # os.makedirs(output_dir, exist_ok=True)
    # process_wav_to_h5_in_directory(estimator_network, root_dir, output_dir)

    # root_dir = f"{DATA_DIR}/ichishima_datas/hen_1_chick_1_7th_time/4w"
    # output_dir = f"{DATA_DIR}/ichishima_datas/BirdAVES_features/hen_1_chick_1_7th_time/4w"

    root_dir = f"{DATA_DIR}/newdata_250626/"
    output_dir = f"{DATA_DIR}/newdata_250626_BirdAVES_features/"
    os.makedirs(output_dir, exist_ok=True)
    process_wav_to_h5_in_directory(estimator_network, root_dir, output_dir)

    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"script begin: {start_time_str}")
    print(f"script end: {end_time_str}")
