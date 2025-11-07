import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Vox(Dataset):
    def __init__(self, config_path, partition, duration_sec, sample_rate, num_examples_per_epoch):
        """ Dataset for vocalization classification with AVES

        Input
        -----
        dataset_dataframe (pandas dataframe): indicating the filepath, annotations and partition of a signal
        is_train (bool): True if in train set
        audio_sr (int): sampling rate expected by network
        duration_sec (float): pad/cut all clips to this duration to enable batching
        annotation_name (str): string corresponding to the annotation columns in the dataframe, e.g. "call_type"
        """
        super().__init__()
        self.data_frame = pd.read_csv(config_path)
        self.dataset_partition = partition
        self.duration_sec = duration_sec
        self.audio_sr = sample_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        
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
          x = x[0, :]
      if sr != self.audio_sr:
          x = torchaudio.functional.resample(x, sr, self.audio_sr)

    #   call_type = row["call_type"]
      return x, row

    # def pad_to_duration(self, x):
    #     """ Pad or clip x to a given duration """
    #     assert len(x.size()) == 1
    #     x_duration = x.size(0) / float(self.audio_sr)
    #     max_samples = int(self.audio_sr * self.duration_sec)
    #     if x_duration == self.duration_sec:
    #         return x
    #     elif x_duration < self.duration_sec:
    #         x = F.pad(x , (0, max_samples - x.size(0)), mode='constant')
    #         return x
    #     else:
    #         return x[:max_samples]

    def __getitem__(self, idx):
        x, row = self.get_one_item(idx)
        out = {"x" : x, "call_type" : row[self.annotation_name]}
        out[self.annotation_name + "_str"] = row[self.annotation_name]
        out[self.annotation_name] = row[self.annotation_name + "_int"]
        return out

# def get_dataloader(dataset_dataframe, is_train, audio_sr, duration_sec, labels, batch_size):
#     return DataLoader(
#             Vox(dataset_dataframe, is_train, audio_sr, duration_sec, labels),
#             batch_size=batch_size,
#             shuffle=is_train,
#             drop_last=is_train
#         )
