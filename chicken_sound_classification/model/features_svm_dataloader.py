import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader

class EmbeddingandLabelDataset(Dataset):
    def __init__(
            self, 
            config_path, 
            partition, 
            # duration_sec, 
            duration_sample,
            duration_embedding,
            # sample_rate, 
            # num_examples_per_epoch
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
        embedding_file = h5py.File(embedding_path, 'r')
        embedding_data = embedding_file["tensor_dataset"][embedding_index, :]
        embedding_tensor = torch.tensor(embedding_data, dtype=torch.float32)
        # embedding_file.close()

        label_path = str(row["label_path"])
        label_file = h5py.File(label_path, 'r')
        label_data = label_file["tensor_dataset"][embedding_index, :]
        label_tensor = torch.tensor(label_data, dtype=torch.float32)
        # label_file.close()

        return row, embedding_tensor, label_tensor

    def __getitem__(self, idx):
        row, embedding, label = self.get_one_item(idx)
        # out = {"x" : signal, "label" : label}
        return embedding, label
    
