import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from features import extract_transport_features

if os.path.exists("data/Augmented_100k_Training_set.csv.gz"):
    train_df = pd.read_csv("data/Augmented_100k_Training_set.csv.gz")
elif os.path.exists("data/Augmented_100k_Training_set.csv"):
    train_df = pd.read_csv("data/Augmented_100k_Training_set.csv")
elif os.path.exists("data/Augmented_1M_Training_set.csv"):
    train_df = pd.read_csv("data/Augmented_1M_Training_set.csv")
else:
    train_df = pd.read_csv("data/Augmented_Training_set.csv")

if len(train_df) > 50000:
    train_df = train_df.sample(n=50000, random_state=42).reset_index(drop=True)

test_df = pd.read_csv("data/Testing_set.csv")
os.makedirs("results", exist_ok=True)

train_fe = extract_transport_features(train_df)
test_fe = extract_transport_features(test_df)

le = LabelEncoder()
y_true = le.fit_transform(train_fe["label"])
class_names = le.classes_
n_classes = len(class_names)
feature_cols = [c for c in train_fe.columns if c not in ["label", "relative_time_0"]]
X = train_fe[feature_cols].values
X_test = test_fe[feature_cols].values

y_permuted = np.random.RandomState(42).permutation(y_true)
clf_null = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=0.1, random_state=42, verbosity=-1, n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
null_preds = np.zeros(len(X))

for tr, val in skf.split(X, y_permuted):
    clf_null.fit(X[tr], y_permuted[tr])
    null_preds[val] = np.argmax(clf_null.predict_proba(X[val]), axis=1)

null_f1 = f1_score(y_permuted, null_preds, average='macro')
null_acc = accuracy_score(y_permuted, null_preds)

seeds = [42, 100, 2026]
fold_scores = []
oof_probs = np.zeros((len(X), n_classes))
test_probs = np.zeros((len(X_test), n_classes))

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, val) in enumerate(skf.split(X, y_true)):
        clf = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.07, min_child_samples=10,
                             colsample_bytree=0.75, subsample=0.85, random_state=seed+fold, verbosity=-1, n_jobs=-1)
        clf.fit(X[tr], y_true[tr])
        val_p = clf.predict_proba(X[val])
        oof_probs[val] += val_p / float(len(seeds))
        f1 = f1_score(y_true[val], np.argmax(val_p, axis=1), average='macro')
        fold_scores.append(f1)
        test_probs += clf.predict_proba(X_test) / (5.0 * float(len(seeds)))

fold_std = float(np.std(fold_scores))
fold_mean = float(np.mean(fold_scores))

overlap_cols = [f"packet_length_{i}" for i in range(5)] + [f"relative_time_{i}" for i in range(1, 5)]
train_tuples = set(tuple(x) for x in train_df[overlap_cols].round(6).values)
test_tuples = set(tuple(x) for x in test_df[overlap_cols].round(6).values)
leakage_overlap_count = len(train_tuples.intersection(test_tuples))

test_entropy = -np.sum(test_probs * np.log2(np.clip(test_probs, 1e-12, 1.0)), axis=1)
mean_entropy = float(np.mean(test_entropy))
low_entropy_pct = float(np.mean(test_entropy < 1.0) * 100.0)

train_preds_full = np.argmax(clf.predict_proba(X), axis=1)
train_f1 = f1_score(y_true, train_preds_full, average='macro')
val_f1 = f1_score(y_true, np.argmax(oof_probs, axis=1), average='macro')
gen_gap = float(train_f1 - val_f1)

print(f"Y-Randomization Null F1: {null_f1:.4f} (Accuracy: {null_acc*100:.2f}%)")
print(f"15-Fold Stability Variance: Mean F1 = {fold_mean:.4f}, Std = {fold_std:.4f}")
print(f"Train-Test Contamination: {leakage_overlap_count} overlapping rows")
print(f"Test Shannon Entropy: Mean = {mean_entropy:.4f} bits, Low Entropy (<1.0 bit) = {low_entropy_pct:.1f}%")
print(f"Train-Val Generalization Gap: Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}, Gap = {gen_gap:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1, len(fold_scores) + 1), fold_scores, marker='o', color='#2563eb', linewidth=2)
axes[0].axhline(fold_mean, color='red', linestyle='--', label=f'Mean F1: {fold_mean:.4f}')
axes[0].fill_between(range(1, len(fold_scores) + 1), fold_mean - fold_std, fold_mean + fold_std, color='#2563eb', alpha=0.15)
axes[0].set_title('15-Fold Validation Stability', fontsize=11)
axes[0].set_xlabel('Fold Index', fontsize=10)
axes[0].set_ylabel('Macro-F1 Score', fontsize=10)
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

sns.histplot(test_entropy, bins=25, kde=True, color='#059669', ax=axes[1])
axes[1].axvline(mean_entropy, color='red', linestyle='--', label=f'Mean Entropy: {mean_entropy:.2f} bits')
axes[1].set_title('Unseen Test Prediction Shannon Entropy Distribution', fontsize=11)
axes[1].set_xlabel('Shannon Entropy (bits)', fontsize=10)
axes[1].set_ylabel('Test Flow Count', fontsize=10)
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("results/overfitting_audit_charts.png", dpi=300, bbox_inches='tight')
plt.close()
