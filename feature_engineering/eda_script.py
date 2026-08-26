import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
os.makedirs("eda_plots", exist_ok=True)
df = pd.read_csv("data/Training_set.csv")
with open("eda_summary.txt", "w") as f:
    f.write("=== Dataset Shape ===\n")
    f.write(f"{df.shape}\n\n")
    f.write("=== Missing Values ===\n")
    f.write(f"{df.isnull().sum().to_string()}\n\n")
    f.write("=== Data Types ===\n")
    f.write(f"{df.dtypes.to_string()}\n\n")
    f.write("=== Summary Statistics ===\n")
    f.write(f"{df.describe().to_string()}\n\n")
    f.write("=== Class Distribution ===\n")
    f.write(f"{df['label'].value_counts().to_string()}\n\n")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='label', order=df['label'].value_counts().index, palette='viridis')
plt.title("Target Class Distribution")
plt.tight_layout()
plt.savefig("eda_plots/class_dist.png")
plt.close()
numeric_df = df.drop(columns=['label'])
plt.figure(figsize=(12, 8))
corr = numeric_df.corr(method='spearman')
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_plots/corr_heatmap.png")
plt.close()
features_to_plot = [f'packet_length_{i}' for i in range(5)]
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(features_to_plot):
    if col in df.columns:
        sns.boxplot(data=df, x=col, y='label', ax=axes[i], showfliers=False)
        axes[i].set_title(col)
        if i > 0:
            axes[i].set_ylabel("")
plt.tight_layout()
plt.savefig("eda_plots/packet_lengths_box.png")
plt.close()
X = numeric_df.values
y = df['label'].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', ax=axes[0], s=30, alpha=0.8)
axes[0].set_title(f"PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})")
axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', ax=axes[1], s=30, alpha=0.8)
axes[1].set_title("t-SNE Projection")
axes[1].legend().remove()
plt.tight_layout()
plt.savefig("eda_plots/pca_tsne.png")
plt.close()
