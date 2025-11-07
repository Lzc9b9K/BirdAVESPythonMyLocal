import os
import torch
import numpy as np
# import pprint
# from argparse import ArgumentParser
import h5py
import shutil

# import torchaudio
# import pytorch_lightning as pl
# import lightning as L
# from lightning.pytorch.loggers import WandbLogger
# from lightning.pytorch.callbacks import ModelCheckpoint

# from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# from aves_svm_network import AVESSVMNetwork
# from aves_svm_dataloader import AudioandLabelDataset
from model.features_svm_network import FeaturesSVMNetwork

SAMPLE_RATE = 16000

# CALL_TYPES = [
#     "foodCall", # Food call
#     "pleasureCall", # Pleasuse call
#     "distressCall", # Distress call
#     "other", # Other
# ]

# AVES_MODEL_PATH = "./content/birdaves-biox-large.torchaudio.pt"
# AVES_MODEL_CONFIG_PATH = "./content/birdaves-biox-large.torchaudio.model_config.json"
DATASET_DIR = "./dataset"

# dummy config
# TRAIN_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241122/four_train_dataset_config.csv"
# VAL_DATASET_CONFIG = "./dataset/nonprocessed_embedding_label_dataset_20241122/four_val_dataset_config.csv"

# PRETRAINED_MODEL_PATH = "./model/pretrained_model/2024-11-25_model_c4ht1rak/epoch=99-train_loss=0.16288-val_loss=0.12690.ckpt"
# PRETRAINED_MODEL_PATH = "./model/pretrained_model/2024-11-28_model_9vo4hh9w/epoch=99-train_loss=0.00014-val_loss=0.00000.ckpt"
# PRETRAINED_MODEL_PATH = "./lightning_logs/qjjcd92s/checkpoints/all_checkpoints/epoch=199-train_loss=0.01014-val_loss=0.00000.ckpt"

DATA_SAMPLES_LENGTH = 960000 # 16000 * 60s * 1
# DATA_SAMPLES_LENGTH = 9600000 # 16000 * 60s * 10
# DATA_EMBEDDINGS_LENGTH = 2999
DATA_SAMPLES_PER_GROUP = 128

def save_aves_label_as_hdf5(tensor_data, output_file):
    # trans
    _, max_indices = torch.max(tensor_data, axis=1)
    aves_label_tensor = torch.zeros_like(tensor_data)
    aves_label_tensor[torch.arange(tensor_data.size(0)), max_indices] = 1

    file_path = output_file
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset("tensor_dataset", data=aves_label_tensor.cpu().detach().numpy())
        # hdf5_file.create_dataset("tensor_dataset", data=tensor_data.cpu().detach().numpy())
    print(f"Save as {file_path}")

def save_time_label_as_hdf5(numpy_data, output_file):
    file_path = output_file
    with h5py.File(file_path, "w") as hdf5_file:
        hdf5_file.create_dataset("tensor_dataset", data=numpy_data)
    print(f"Save as {file_path}")

def add_file_aves_label_suffix_name(filename):
    base_name, ext = os.path.splitext(filename)
    if ext.lower() == ".h5":
        new_filename = f"{base_name}_aves_label.h5"
        return new_filename
    else:
        return None
    
def add_file_time_label_suffix_name(filename):
    base_name, ext = os.path.splitext(filename)
    if ext.lower() == ".h5":
        new_filename = f"{base_name}_time.h5"
        return new_filename
    else:
        return None

def output_aves_label(estimator_network, embedding_dir, output_dir):
    """
        from embeddings file to aves dim label file
    """
    for dirpath, dirnames, filenames in os.walk(embedding_dir):
        for filename in filenames:
            if filename.lower().endswith('.h5'):
                # load embeddings file
                embedding_file_path = os.path.join(dirpath, filename)
                x = h5py.File(embedding_file_path, "r")
                x_data = x["tensor_dataset"]
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
                # output labels
                l = estimator_network(x_tensor)
                l_squeezed = l.squeeze()
                # save labels hdf5
                l_file_name = add_file_aves_label_suffix_name(filename)
                relative_path = os.path.relpath(dirpath, embedding_dir)
                target_dir = os.path.join(output_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                l_file_save_path = os.path.join(target_dir, l_file_name)
                # save
                save_aves_label_as_hdf5(l_squeezed, l_file_save_path)
                # close hdf5 file
                x.close()


def avesDim_to_timeDim(aves_label_numpy):
    expanded_label = np.repeat(
        aves_label_numpy, 
        DATA_SAMPLES_PER_GROUP, 
        axis=0,
        )
    num_rows_to_fill = DATA_SAMPLES_LENGTH - expanded_label.shape[0]
    if num_rows_to_fill > 0:
        last_row = expanded_label[-1]
        fill_rows = np.tile(last_row, (num_rows_to_fill, 1))
        expanded_label = np.vstack((expanded_label, fill_rows))
    
    return expanded_label 

def avesDim_label_to_timeDim_label(aves_label_dir, output_dir):
    """
        from aves dim label to time dim label
    """
    for dirpath, dirnames, filenames in os.walk(aves_label_dir):
        for filename in filenames:
            if filename.lower().endswith('.h5'):
                # load embeddings file
                embedding_file_path = os.path.join(dirpath, filename)
                x = h5py.File(embedding_file_path, "r")
                x_data = x["tensor_dataset"]
                x_aves_array = np.array(x_data)

                # process
                x_time = avesDim_to_timeDim(x_aves_array)

                # save labels hdf5
                l_file_name = add_file_time_label_suffix_name(filename)
                relative_path = os.path.relpath(dirpath, aves_label_dir)
                target_dir = os.path.join(output_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)
                l_file_save_path = os.path.join(target_dir, l_file_name)
                # save
                save_time_label_as_hdf5(x_time, l_file_save_path)
                # close hdf5 file
                x.close()
    shutil.rmtree(aves_label_dir)
    return 


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    # L.seed_everything(42)

    # path

    # band passed
    embedding_dir = f"{DATASET_DIR}/2025_stft_dataset/bandPassed/test_dataset/embeddings"
    # 2 labels
    pretrained_model_path = "2025_trained_model/stft_1layer_model_with_softmax/2_bandPassed/uyjzb6eo/checkpoints/all_checkpoints/epoch=99-train_loss=0.31326.ckpt"
    data_saving_dir = f"{DATASET_DIR}/2025_stft_1layer_mode_with_softmax"
    aves_label_dir = f"{data_saving_dir}/full_avex_label"
    time_label_dir = f"{data_saving_dir}/new_bp_2_full_time_label_ep_99"
    # create model
    estimator_network = FeaturesSVMNetwork.load_from_checkpoint(pretrained_model_path)
    estimator_network.eval()
    # output label
    output_aves_label(estimator_network, embedding_dir, aves_label_dir)
    avesDim_label_to_timeDim_label(aves_label_dir, time_label_dir)

    # # original passed
    embedding_dir = f"{DATASET_DIR}/2025_stft_dataset/original/test_dataset/embeddings"
    # 2 labels
    pretrained_model_path = "2025_trained_model/stft_3layer_model/2_original/910unn6b/checkpoints/all_checkpoints/epoch=99-train_loss=0.31326.ckpt"
    data_saving_dir = f"{DATASET_DIR}/2025_stft_3layer_mode_with_softmax"
    aves_label_dir = f"{data_saving_dir}/full_avex_label"
    time_label_dir = f"{data_saving_dir}/o_2_full_time_label_ep_99"
    # create model
    # estimator_network = FeaturesSVMNetwork.load_from_checkpoint(pretrained_model_path)
    # estimator_network.eval()
    # # output label
    # output_aves_label(estimator_network, embedding_dir, aves_label_dir)
    # avesDim_label_to_timeDim_label(aves_label_dir, time_label_dir)
    # # # 4 labels
    # pretrained_model_path = "2025_trained_model/3layer_model/2_original/w625m92p/checkpoints/all_checkpoints/epoch=199-train_loss=0.00048.ckpt"
    # data_saving_dir = f"{DATASET_DIR}/2025_3layer_mode_with_softmax"
    # aves_label_dir = f"{data_saving_dir}/full_avex_label"
    # time_label_dir = f"{data_saving_dir}/o_4_full_time_label_ep_199"
    # # create model
    # estimator_network = FeaturesSVMNetwork.load_from_checkpoint(pretrained_model_path)
    # estimator_network.eval()
    # # output label
    # output_aves_label(estimator_network, embedding_dir, aves_label_dir)
    # avesDim_label_to_timeDim_label(aves_label_dir, time_label_dir)

    # high passed
    # embedding_dir = f"{DATASET_DIR}/20241225/highPassedDatas"
    # # 2 labels
    # # pretrained_model_path = "2025_trained_model/3layer_model/2_highPassed/gydk9hhe/checkpoints/all_checkpoints/epoch=499-train_loss=0.00011.ckpt"
    # pretrained_model_path = "2025_trained_model/3layer_model_with_softmax/2_highPassed/ku7wn84z/checkpoints/all_checkpoints/epoch=99-train_loss=0.31564.ckpt"
    # data_saving_dir = f"{DATASET_DIR}/2025_3layer_mode_with_softmax"
    # aves_label_dir = f"{data_saving_dir}/full_avex_label"
    # time_label_dir = f"{data_saving_dir}/hp_2_full_time_label_ep_99"
    # # create model
    # estimator_network = FeaturesSVMNetwork.load_from_checkpoint(pretrained_model_path)
    # estimator_network.eval()
    # # output label
    # output_aves_label(estimator_network, embedding_dir, aves_label_dir)
    # avesDim_label_to_timeDim_label(aves_label_dir, time_label_dir)
    # # 4 labels
    # pretrained_model_path = "2025_trained_model/3layer_model/2_highPassed/gydk9hhe/checkpoints/all_checkpoints/epoch=99-train_loss=0.31715.ckpt"
    # data_saving_dir = f"{DATASET_DIR}/2025_3layer_mode_with_softmax"
    # aves_label_dir = f"{data_saving_dir}/full_avex_label"
    # time_label_dir = f"{data_saving_dir}/hp_4_full_time_label_ep_499"
    # # # create model
    # # estimator_network = FeaturesSVMNetwork.load_from_checkpoint(pretrained_model_path)
    # # estimator_network.eval()
    # # # output label
    # # output_aves_label(estimator_network, embedding_dir, aves_label_dir)
    # # avesDim_label_to_timeDim_label(aves_label_dir, time_label_dir)
