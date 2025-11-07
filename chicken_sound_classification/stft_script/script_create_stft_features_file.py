import os
import shutil
import scipy.io.wavfile as wav
import random
from datetime import datetime

import h5py
import numpy as np

import torch
import torchaudio

import librosa

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
        hdf5_file.create_dataset("tensor_dataset", data=tensor_data)
    print(f"Save as {file_path}")

def replace_wav_with_h5(filename):
    base_name, ext = os.path.splitext(filename)
    if ext.lower() == ".wav":
        new_filename = f"{base_name}.h5"
        return new_filename
    else:
        return None
    
def process_wav_to_h5_in_directory(root_directory, output_root_directory):
    n_fft = 1024
    hop_length = 128

    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                # torch load
                # print(f"dirpath: {dirpath}")
                # print(f"dirnames: {dirnames}")
                # print(f"filename: {filename}")
                audio_file_path = os.path.join(dirpath, filename)
                # x, sr = torchaudio.load(audio_file_path)
                x, sr = librosa.load(audio_file_path, sr=None)
                # print(f"sr: {sr}")

                # model processing
                # output_features, _ = estimator_network(x)
                stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
                # magnitude = np.abs(stft)
                magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
                output_features = magnitude_db.T
                # print(f"stft: {stft.shape}")
                print(f"output_features: {output_features.shape}")
                # output_features_squeezed = output_features.squeeze()

                # save hdf5
                hdf5_file_name = replace_wav_with_h5(filename)
                relative_path = os.path.relpath(dirpath, root_directory)
                # print(f"relative_path : {relative_path}")
                # hdf5_file_save_path = f"{output_root_directory}/{relative_path}/{hdf5_file_name}"
                target_dir = os.path.join(output_root_directory, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                hdf5_file_save_path = os.path.join(target_dir, hdf5_file_name)
                # save
                save_as_hdf5(output_features, hdf5_file_save_path)

if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    # torch.backends.cudnn.benchmark = True
    # L.seed_everything(42)

    # estimator_network = AVESSVMNetwork(
    #     aves_config_path=AVES_MODEL_CONFIG_PATH, 
    #     aves_model_path=AVES_MODEL_PATH,
    #     aves_trainable=False,
    #     n_classes=4,
    #     train_dataset_config=TRAIN_DATASET_CONFIG,
    #     test_dataset_config=VAL_DATASET_CONFIG,
    #     batch_size=32,
    #     learning_rate=1e-3,
    #     embedding_dim=1024,
    #     audio_sr=16000,
    # )
    # estimator_network.eval()

    root_dir = f"{DATA_DIR}/2025_stft_dataset/bandPassed/test_dataset"
    output_dir = f"{DATA_DIR}/2025_stft_dataset/bandPassed/test_dataset/embeddings"
    os.makedirs(output_dir, exist_ok=True)
    process_wav_to_h5_in_directory(root_dir, output_dir)

    # root_dir = f"{DATA_DIR}/20241205_highPassed_dataset"
    # output_dir = f"{DATA_DIR}/20241205_highPassed_dataset"
    # os.makedirs(output_dir, exist_ok=True)
    # process_wav_to_h5_in_directory(estimator_network, root_dir, output_dir)

    # root_dir = f"{DATA_DIR}/20241205_bandPassed_dataset"
    # output_dir = f"{DATA_DIR}/20241205_bandPassed_dataset"
    # os.makedirs(output_dir, exist_ok=True)
    # process_wav_to_h5_in_directory(estimator_network, root_dir, output_dir)
