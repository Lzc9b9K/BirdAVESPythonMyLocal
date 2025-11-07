import os
import scipy.io.wavfile as wav
import random
from datetime import datetime

import numpy as np

import torch
import torchaudio

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from aves_svm_network import AVESSVMNetwork

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

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    
    torch.backends.cudnn.benchmark = True
    L.seed_everything(42)

    # create different length data
    # full_audio_path = "./dataset/total/total_signal.wav"
    # sample_rate, data = wav.read(full_audio_path)
    # segment_lengths = [300, 600, 720, 900, 1200]  # 10s, 30s, 1min, 2min, 3min
    # output_dir = './dataset/audio_segments'
    # os.makedirs(output_dir, exist_ok=True)
    # for length in segment_lengths:
    #     num_samples = sample_rate * length
    #     start_index = random.randint(0, len(data) - num_samples)
    #     segment = data[start_index:start_index + num_samples]
    #     # save
    #     output_file = os.path.join(output_dir, f'segment_{length}s.wav')
    #     wav.write(output_file, sample_rate, segment)
    #     print(f'Saved: {output_file}')

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

    # 16 mins file processing
    # audio_file_16mins_path = "./dataset/total/highpass_150_chicken_sound_16mins.wav"
    # x, sr = torchaudio.load(audio_file_16mins_path)
    # print(f"test begin: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
    # labels_16mins = estimator_network(x)
    # print(f"test end: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
    # print(f"16 mins audio labels: {labels_16mins.size()}")
    

    # 20 mins file processing
    # audio_file_20mins_path = "./dataset/total/total_signal.wav"
    # x, sr = torchaudio.load(audio_file_20mins_path)
    # print(f"test begin: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
    # labels_20mins = estimator_network(x)
    # print(f"test end: {datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}")
    # print(f"20 mins audio labels: {labels_20mins.size()}")

    length = "1200"
    audio_file_path = f"./dataset/audio_segments/segment_{length}s.wav"
    x, sr = torchaudio.load(audio_file_path)
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"test begin: {start_time_str}")
    output_labels = estimator_network(x)
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"test end: {end_time_str}")
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    print(f"running time cost: {elapsed_seconds} s")
    print(f"{length} s audio labels: {output_labels.size()}")
