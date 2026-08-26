import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from advanced_feature_engineering_and_selection import extract_super_features
os.makedirs("eda_plots_augmented", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
df_raw = pd.read_csv("data/Augmented_Training_set.csv")
df_full = extract_super_features(df_raw)
feat_cols = [c for c in df_full.columns if c != 'label']
X = df_full[feat_cols]
y = df_full['label']
le = LabelEncoder()
y_enc = le.fit_transform(y)
class_names = le.classes_
mi_scores = mutual_info_classif(X, y_enc, random_state=42)
df_mi = pd.DataFrame({'Feature': feat_cols, 'Mutual_Information': mi_scores}).sort_values('Mutual_Information', ascending=False)
clf = ExtraTreesClassifier(n_estimators=300, max_depth=12, random_state=42)
clf.fit(X, y_enc)
df_mi['ExtraTrees_Importance'] = clf.feature_importances_
plt.figure(figsize=(12, 8))
top_25 = df_mi.head(25).sort_values('Mutual_Information', ascending=True)
bars = plt.barh(top_25['Feature'], top_25['Mutual_Information'], color='
plt.title('Top 25 Most Discriminative Features (Mutual Information Ranking)', fontsize=14, fontweight='bold', color='
plt.xlabel('Mutual Information Score (Bits of Information)', fontsize=11, fontweight='bold')
plt.ylabel('Feature Name', fontsize=11, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('eda_plots_augmented/top_25_features_mutual_info.png', dpi=300)
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df_full, x='label', y='harmonic_resonance_20ms', ax=axes[0], palette='tab10')
axes[0].set_title('Multi-Harmonic 20ms Opus Resonance Score by Class', fontsize=12, fontweight='bold', color='
axes[0].set_xlabel('Traffic Class', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Resonance Score: exp(-dist / sigma)', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
sns.boxplot(data=df_full, x='label', y='trajectory_curvature_a', ax=axes[1], palette='tab10')
axes[1].set_title('Polynomial Trajectory Curvature (a) by Class', fontsize=12, fontweight='bold', color='
axes[1].set_xlabel('Traffic Class', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Curvature Parameter (a)', fontsize=10, fontweight='bold')
axes[1].set_ylim(-200000, 200000)
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('eda_plots_augmented/harmonic_and_curvature_separation.png', dpi=300)
plt.close()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=df_full, x='label', y='burst_duty_cycle', ax=axes[0], palette='viridis', ci=None, edgecolor='black')
axes[0].set_title('Sub-Millisecond Burst Duty Cycle by Class', fontsize=12, fontweight='bold', color='
axes[0].set_xlabel('Traffic Class', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Fraction of Time in Burst Occupancy', fontsize=10, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(data=df_full, x='label', y='shannon_len_entropy', ax=axes[1], palette='magma', ci=None, edgecolor='black')
axes[1].set_title('Shannon Packet Length Entropy by Class', fontsize=12, fontweight='bold', color='
axes[1].set_xlabel('Traffic Class', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Length Entropy (Bits)', fontsize=10, fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('eda_plots_augmented/duty_cycle_and_entropy.png', dpi=300)
plt.close()
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_full,
    x='log_energy_mean',
    y='dlen_1',
    hue='label',
    palette='tab10',
    alpha=0.7,
    s=40
)
plt.title('Log-Energy Spectrum vs First-Order Length Delta (dlen_1)', fontsize=13, fontweight='bold', color='
plt.xlabel('Mean Log-Energy: mean(log(L^2 + 1))', fontsize=11, fontweight='bold')
plt.ylabel('Initial Length Jump (dlen_1 in Bytes)', fontsize=11, fontweight='bold')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('eda_plots_augmented/log_energy_vs_delta_scatter.png', dpi=300)
plt.close()
with open("eda_plots_augmented/eda_summary.txt", "w") as f:
    f.write("=== TOP 25 FEATURES BY MUTUAL INFORMATION ===\n")
    f.write(df_mi.head(25).to_string(index=False))
