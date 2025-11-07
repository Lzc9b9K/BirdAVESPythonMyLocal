import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

# features_10min_unsqueeze = torch.load('./results/chicken_sound_features_10min.pt')
# features_6min_unsqueeze = torch.load('./results/chicken_sound_features_6min.pt')
features_10min_unsqueeze = torch.load('./results/ABB_chicken_sound_features_10min.pt')
features_6min_unsqueeze = torch.load('./results/ABB_chicken_sound_features_6min.pt')
features_10min = features_10min_unsqueeze.squeeze(0)
features_6min = features_6min_unsqueeze.squeeze(0)
# print("features_10min.size()", features_10min.size())
# print("features_6min.size()", features_6min.size())

labels_10min_df = pd.read_csv("./labels/compressed_10min.csv")
labels_6min_df = pd.read_csv("./labels/compressed_6min.csv")
# print("labels_10min_df.shape", labels_10min_df.shape)
# print("labels_6min_df.shape", labels_6min_df.shape)

valid_indices_10min = labels_10min_df.sum(axis=1) > 0  # 找到有效行
labels_10min = labels_10min_df.idxmax(axis=1)  # 获取每行中值为1的列名
labels_10min = labels_10min[valid_indices_10min]  # 过滤掉无效行
features_10min = features_10min[valid_indices_10min]  # 过滤掉无效行的特征

valid_indices_6min = labels_6min_df.sum(axis=1) > 0  # 找到有效行
labels_6min = labels_6min_df.idxmax(axis=1)  # 获取每行中值为1的列名
labels_6min = labels_6min[valid_indices_6min]  # 过滤掉无效行
features_6min = features_6min[valid_indices_6min]  # 过滤掉无效行的特征

combined_features = torch.cat((features_10min, features_6min), dim=0)
print(f"combined_features: {combined_features.shape}")
combined_labels = pd.concat([labels_10min, labels_6min], axis=0).reset_index(drop=True)

tsne = TSNE(n_components=2, random_state=0, perplexity=30)
features_2d = tsne.fit_transform(combined_features.detach().numpy())
print(f"combined_features.detach().numpy(): {combined_features.detach().numpy().shape}")

unique_labels = combined_labels.unique()
colors = plt.cm.get_cmap('tab10', len(unique_labels))
plt.figure(figsize=(10, 8))
for i, label in enumerate(unique_labels):
    indices = combined_labels == label
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label, color=colors(i))
plt.legend()
plt.title('t-SNE Visualization of Combined Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.savefig('tsne_combined.png', dpi=300)
plt.show()
