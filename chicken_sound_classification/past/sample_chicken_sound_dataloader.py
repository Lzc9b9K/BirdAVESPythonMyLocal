import torch
import torchaudio
# import torch.nn.functional as F
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader

# import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
# import soundfile as sf
from sklearn.model_selection import train_test_split

# learning_sample
LEARNING_SAMPLE = 1600

class AudioDataset(Dataset):
    def __init__(self, audio_file, label_file, sample_rate=16000, segment_length=16000):
        self.audio, self.sr = torchaudio.load(audio_file)
        # print(f"self.audio: {self.audio}")
        # print(f"self.audio.length: {len(self.audio)}")
        self.labels = pd.read_csv(label_file).values
        self.segment_length = segment_length
        if len(self.audio.size()) == 2:
          self.audio = self.audio[0, :]
        self.num_segments = len(self.audio) // self.segment_length
        self.sample_rate = sample_rate
        self.learning_parts = self.sample_rate / LEARNING_SAMPLE
    def __len__(self):
        return self.num_segments
    def __getitem__(self, idx):
        start_sample = idx * self.segment_length
        end_sample = start_sample + self.segment_length

        # audio
        audio_segment = self.audio[start_sample:end_sample]
        # print(f"audio_segment: {audio_segment}")
        # audio_segment = audio_segment.unsqueeze(1)
        reshaped_audio_segment = audio_segment.view(self.learning_parts, LEARNING_SAMPLE)
        # print(f"after audio_segment: {audio_segment}")

        # label
        label_segment = self.labels[start_sample:end_sample]
        labels_array = np.array(label_segment)
        # print(f"label segment: {label_segment}")
        # print(f"label segment length: {len(label_segment)}")
        # print(f"label labels_array: {labels_array}")
        reshaped_labels =  labels_array.reshape([self.learning_parts, LEARNING_SAMPLE, 4])
        # print(f"reshaped_labels : {reshaped_labels}")
        compressed_labels = np.zeros((self.learning_parts, 4), dtype=int)
        for i in range(self.learning_parts):
            for j in range(4):
                counts = np.bincount(reshaped_labels[i, :, j])
                compressed_labels[i, j] = np.argmax(counts)
        return reshaped_audio_segment, compressed_labels


audio_file = 'dataset/total/total_signal.wav'
label_file = 'dataset/total/total_label_type1.csv'
dataset = AudioDataset(audio_file, label_file)

indices = np.arange(len(dataset))
labels = [np.argmax(dataset[i][1].sum(axis=0)) for i in indices] 
# print(f"indices: {indices}")
# print(f"labels: {labels}")

# train_indices, temp_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
# print(f"train_indices: {train_indices}")
# print(f"temp_indices: {temp_indices}")
# print(f"[labels[i] for i in temp_indices]: {[labels[i] for i in temp_indices]}")
# val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, stratify=[labels[i] for i in temp_indices], random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
