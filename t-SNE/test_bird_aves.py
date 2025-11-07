import fairseq
import torch
import torch.nn as nn
import torchaudio

class AvesClassifier(nn.Module):
    def __init__(self, model_path, num_classes, embeddings_dim=1024, multi_label=False):

        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # print(self.model.extract_features(x).shape())
        
        extract_features = self.model.extract_features(x)
        # print(f"extract_features(x)) : {extract_features}")
        # print(f"len(extract_features(x)) : {len(extract_features)}")
        origin_out = extract_features[0]
        print(f"origin_out.size() : {origin_out.size()}")
        out = origin_out.mean(dim=1)  # mean pooling
        print(f"out.size() : {out.size()}")
        logits = self.head(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return origin_out, loss, logits


# Initialize an AVES classifier with 10 target classes
model = AvesClassifier(
    model_path='./ckpts/birdaves-bioxn-large.pt',
    # model_path='./ckpts/aves-base-bio.pt',
    num_classes=4)

model.eval()

# Create a 1-second random sound
# waveform = torch.rand((16_000))
# x = waveform.unsqueeze(0)
# x, sr = torchaudio.load("./audios/chicken_sound.wav")
# x, sr = torchaudio.load("./audios/chicken_sound.wav", frame_offset=16_000, num_frames=160_000)
# x, sr = torchaudio.load("./audios/chicken_sound_10.wav")
# x, sr = torchaudio.load("./audios/chicken_sound_6.wav")
# print("sr:", sr)

y = torch.tensor([0])

# Run the forward pass
# x1, sr = torchaudio.load("./audios/chicken_sound_10.wav")
# out1, loss, logits = model(x1, y)
# torch.save(out1, "./results/ABB_chicken_sound_features_10min.pt")

# x2, sr = torchaudio.load("./audios/chicken_sound_6.wav")
# out2, loss, logits = model(x2, y)
# 1s data
# x, sr = torchaudio.load("./chicken_sound_classification/dataset/16mins_1s_dataset/train/Fo_00_04_24_729.wav")
# out, loss, logits = model(x, y)
# # 0.1s data
# x, sr = torchaudio.load("./chicken_sound_classification/dataset/20mins_01s_dataset/train/Di_00_02_38_841.wav")
# out, loss, logits = model(x, y)
# # 0.2s data
# x, sr = torchaudio.load("./chicken_sound_classification/dataset/20mins_02s_dataset/train/Fo_00_17_47_356.wav")
# out, loss, logits = model(x, y)