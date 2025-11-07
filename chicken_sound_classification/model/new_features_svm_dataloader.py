import torch
import torchaudio
import h5py
import pandas as pd
import numpy as np
from scipy.stats import mode
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TRAIN_SAMPLES_PER_GROUP = 320

class EmbeddingandLabelDataset(Dataset):
    def __init__(
            self, 
            config_path, 
            partition, 
            # duration_sec, 
            duration_sample,
            duration_embedding,
            # sample_rate, 
            # num_examples_per_epoch,
            # device,
        ):
        """ Dataset for vocalization classification with AVES

        Input
        -----
        dataset_dataframe (pandas dataframe): indicating the filepath, annotations and partition of a signal
        is_train (bool): True if in train set
        audio_sr (int): sampling rate expected by network
        # duration_sec (float): pad/cut all clips to this duration to enable batching
        annotation_name (str): string corresponding to the annotation columns in the dataframe, e.g. "call_type"
        """
        super().__init__()
        self.data_frame = pd.read_csv(config_path)
        self.dataset_partition = partition
        # self.duration_sec = duration_sec
        # self.audio_sr = sample_rate
        # self.num_examples_per_epoch = num_examples_per_epoch
        self.duration_sample = duration_sample
        self.duration_embedding = duration_embedding
        # self.run_device = device
        
        # self.annotation_name = "call_type"
        # class_annotations = pd.Categorical(self.data_frame[self.annotation_name])
        # self.classes = class_annotations.categories
        # dataset_dataframe = self.data_frame.copy()
        # self.data_frame[self.annotation_name + "_int"] = class_annotations.codes
        # print(self.data_frame)
        self.dataset_info = self.data_frame[self.data_frame["subset"] == self.dataset_partition]

    def __len__(self):
        return len(self.dataset_info)

    def get_one_item(self, idx):
        """ Load base audio """
        row = self.data_frame.iloc[idx]
        # x, sr = torchaudio.load(row["filepath"])
        # if len(x.size()) == 2:
        #     signal = x[0, :]
        # if sr != self.audio_sr:
        #     signal = torchaudio.functional.resample(x, sr, self.audio_sr)
        
        embedding_index = int(row["embedding_index"])
        embedding_path = str(row["embedding_path"])
        label_path = str(row["label_path"])

        with h5py.File(embedding_path, 'r') as embedding_file:
            # embedding_file = h5py.File(embedding_path, 'r')
            embedding_data = embedding_file["tensor_dataset"][embedding_index, :]
            embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
            # embedding_file.close()
        with h5py.File(label_path, 'r') as label_file:
            # label_path = str(row["label_path"])
            # label_file = h5py.File(label_path, 'r')
            label_data = label_file["tensor_dataset"][embedding_index, :]
            label_tensor = torch.tensor(label_data, dtype=torch.float32)
            # label_file.close()  
          
        # return row, embedding_tensor.to(self.run_device), label_tensor.to(self.run_device)
        return row, embedding_tensor, label_tensor

    def __getitem__(self, idx):
        row, embedding, label = self.get_one_item(idx)
        # out = {"x" : signal, "label" : label}
        return {}, embedding, label

class ValidationEmbeddingandLabelDataset(Dataset):
    def __init__(
            self, 
            config_path, 
            partition, 
            classes_nums,
        ):
        """ Dataset for vocalization classification with AVES

        Input
        -----
        dataset_dataframe (pandas dataframe): indicating the filepath, annotations and partition of a signal
        is_train (bool): True if in train set
        audio_sr (int): sampling rate expected by network
        # duration_sec (float): pad/cut all clips to this duration to enable batching
        annotation_name (str): string corresponding to the annotation columns in the dataframe, e.g. "call_type"
        """
        super().__init__()
        self.data_frame = pd.read_csv(config_path)
        self.dataset_partition = partition
        # self.num_examples_per_epoch = num_examples_per_epoch
        self.classes_nums = classes_nums
        
        # self.annotation_name = "call_type"
        # class_annotations = pd.Categorical(self.data_frame[self.annotation_name])
        # self.classes = class_annotations.categories
        self.dataset_info = self.data_frame[self.data_frame["subset"] == self.dataset_partition]

    def __len__(self):
        return len(self.dataset_info)

    def get_one_item(self, idx):
        """ Load base audio """
        row = self.data_frame.iloc[idx]
        
        # embedding_index = int(row["embedding_index"])
        features_file = str(row["features_file"])
        labels_file = str(row["labels_file"])
        labels_dataset = str(row["labels_dataset"])
        # recording_number = str(row["recording_number"])
        # age = str(row["age"])
        attr_dict = {
            "recording_number": str(row["recording_number"]),
            "age": str(row["age"]),
        }

        with h5py.File(features_file, 'r') as features_file_data:
            embedding_data = features_file_data["tensor_dataset"]
            embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)

        with h5py.File(labels_file, 'r') as labels_file_data:
            label_data = labels_file_data[labels_dataset]
            label_tensor = torch.tensor(label_data, dtype=torch.float32)
        label_tensor = self.compress_labels(label_tensor, embedding_tensor, self.classes_nums)
        # print("\n label_tensor.shape", label_tensor.shape, flush=True)

        return attr_dict, embedding_tensor, label_tensor

    def __getitem__(self, idx):
        attr_dict, embedding, label = self.get_one_item(idx)

        return attr_dict, embedding, label

    @staticmethod
    def compress_labels(labels_tensor, embedding_tensor, classes_num):
        # print("labels_tensor.shape", labels_tensor.shape, flush=True)
        # print("embedding_tensor.shape", embedding_tensor.shape, flush=True)
        num_groups = len(labels_tensor) // TRAIN_SAMPLES_PER_GROUP
        compressed_labels = []
        
        for i in range(num_groups):
            start_sample = i * TRAIN_SAMPLES_PER_GROUP
            end_sample = start_sample + TRAIN_SAMPLES_PER_GROUP
            group_labels = labels_tensor[start_sample:end_sample]

            group_mode = mode(group_labels, axis=0).mode[0]
            compressed_labels.append(group_mode)
        
        compressed_labels_tensor = torch.tensor(compressed_labels)
        embedding_length = len(embedding_tensor)
        if len(compressed_labels_tensor) > embedding_length:
            compressed_labels_tensor = compressed_labels_tensor[:embedding_length]
        elif len(compressed_labels_tensor) < embedding_length:
            padding_length = embedding_length - len(compressed_labels_tensor)
            padding = compressed_labels_tensor[-1].repeat(padding_length, 1)
            compressed_labels_tensor = torch.cat((compressed_labels_tensor, padding), dim=0)
        # print("compressed_labels_tensor.shape", compressed_labels_tensor.shape, flush=True)

        foodCall = compressed_labels_tensor[:, 0]
        # print("foodCall.shape", foodCall.shape)
        pleasureCall = compressed_labels_tensor[:, 1].bool()
        distressCall = compressed_labels_tensor[:, 2].bool()
        fearTrill = compressed_labels_tensor[:, 3].bool()
        other = compressed_labels_tensor[:, 4].bool()
        other = other | fearTrill

        compressed_labels_tensor = torch.stack([foodCall, pleasureCall, distressCall, other], dim=1)
        one_hot_count = compressed_labels_tensor.sum(dim=1)
        mask = one_hot_count > 1
        foodCall[mask] = 0
        pleasureCall[mask] = 0
        distressCall[mask] = 0
        other[mask] = 1
        if classes_num == 4:
            compressed_labels_tensor = torch.stack([foodCall, pleasureCall, distressCall, other], dim=1)
        elif classes_num == 2:
            other = other | distressCall | pleasureCall
            compressed_labels_tensor = torch.stack([foodCall, other], dim=1)

        return compressed_labels_tensor

    # @staticmethod
    # def expand_labels(labels_tensor):
    #     return 
