import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class AudioandLabelDataset(Dataset):
    def __init__(
            self, 
            config_path, 
            partition, 
            # duration_sec, 
            sample_rate, 
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
        self.audio_sr = sample_rate
        # self.num_examples_per_epoch = num_examples_per_epoch
        
        self.annotation_name = "call_type"
        class_annotations = pd.Categorical(self.data_frame[self.annotation_name])
        self.classes = class_annotations.categories
        # dataset_dataframe = self.data_frame.copy()
        self.data_frame[self.annotation_name + "_int"] = class_annotations.codes
        # print(self.data_frame)
        self.dataset_info = self.data_frame[self.data_frame["subset"] == self.dataset_partition]

    def __len__(self):
        return len(self.dataset_info)

    def get_one_item(self, idx):
        """ Load base audio """
        row = self.data_frame.iloc[idx]
        x, sr = torchaudio.load(row["filepath"])
        if len(x.size()) == 2:
            signal = x[0, :]
        if sr != self.audio_sr:
            signal = torchaudio.functional.resample(x, sr, self.audio_sr)
          
        label_df = pd.read_csv(row["labelpath"])
        label_array = label_df.values
        label_tensor = torch.tensor(label_array, dtype=torch.float32)

        return row, signal, label_tensor

    def __getitem__(self, idx):
        row, signal, label = self.get_one_item(idx)
        # out = {"x" : signal, "label" : label}
        return signal, label
