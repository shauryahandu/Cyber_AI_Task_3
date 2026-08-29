import time
import warnings
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import skew, kurtosis, entropy
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from features import extract_transport_features, build_sequence_tensor
from model import BiGRUEmbedder

warnings.filterwarnings('ignore')
os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists("data/Augmented_1M_Training_set.csv"):
    train_raw = pd.read_csv("data/Augmented_1M_Training_set.csv")
else:
    train_raw = pd.read_csv("data/Augmented_Training_set.csv")
test_raw = pd.read_csv("data/Testing_set.csv")

train_fe = extract_transport_features(train_raw)
test_fe = extract_transport_features(test_raw)

le_all = LabelEncoder()
y_enc = le_all.fit_transform(train_fe["label"])
class_names = le_all.classes_
n_classes = len(class_names)
N = len(train_raw)

sample_sub = train_fe.sample(n=min(50000, N), random_state=42)
sample_y = le_all.transform(sample_sub["label"])
raw_feats = [c for c in train_fe.columns if c not in ["label", "relative_time_0"]]
corr_matrix = pd.DataFrame(sample_sub[raw_feats].values, columns=raw_feats).corr(method='spearman').abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [c for c in upper_tri.columns if any(upper_tri[c] > 0.98)]
non_coll_feats = [c for c in raw_feats if c not in drop_cols]

mi_scores = mutual_info_classif(sample_sub[non_coll_feats].values, sample_y, random_state=42)
clf_lgb_s = LGBMClassifier(n_estimators=250, num_leaves=31, learning_rate=0.08, random_state=42, verbosity=-1, n_jobs=-1)
clf_lgb_s.fit(sample_sub[non_coll_feats].values, sample_y)
imp_comb = (0.5 * (mi_scores / mi_scores.max()) + 0.5 * (clf_lgb_s.feature_importances_ / clf_lgb_s.feature_importances_.max()))
selected_feats = pd.Series(imp_comb, index=non_coll_feats).sort_values(ascending=False).head(54).index.tolist()

X_seq_train = build_sequence_tensor(train_raw)
X_seq_test = build_sequence_tensor(test_raw)

gru_model = BiGRUEmbedder().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(gru_model.parameters(), lr=0.002, weight_decay=1e-4)

train_dataset = TensorDataset(torch.tensor(X_seq_train, dtype=torch.float32), torch.tensor(y_enc, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

gru_model.train()
for epoch in range(20):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        _, logits = gru_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

gru_model.eval()
with torch.no_grad():
    train_emb, _ = gru_model(torch.tensor(X_seq_train, dtype=torch.float32).to(device))
    test_emb, _ = gru_model(torch.tensor(X_seq_test, dtype=torch.float32).to(device))
    train_emb = train_emb.cpu().numpy()
    test_emb = test_emb.cpu().numpy()

X_fused_train = np.hstack([train_fe[selected_feats].values, train_emb])
X_fused_test = np.hstack([test_fe[selected_feats].values, test_emb])

voice_classes = [c for c in class_names if "voice" in c]
video_classes = [c for c in class_names if "video" in c]

le_voice = LabelEncoder()
le_voice.fit(voice_classes)
le_video = LabelEncoder()
le_video.fit(video_classes)

y_modality = np.array([1 if "video" in c else 0 for c in train_fe["label"]])

voice_mask_all = (y_modality == 0)
video_mask_all = (y_modality == 1)

y_enc_voice = le_voice.transform(train_fe.loc[voice_mask_all, "label"])
y_enc_video = le_video.transform(train_fe.loc[video_mask_all, "label"])

seeds = [42, 100, 2026]
oof_router_probs = np.zeros(N)
test_router_probs = np.zeros(len(test_raw))

for seed in seeds:
    skf_m = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, val) in enumerate(skf_m.split(X_fused_train, y_modality)):
        clf_r = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.80, random_state=seed+fold, verbosity=-1, n_jobs=-1)
        clf_r.fit(X_fused_train[tr], y_modality[tr])
        oof_router_probs[val] += clf_r.predict_proba(X_fused_train[val])[:, 1] / float(len(seeds))
        test_router_probs += clf_r.predict_proba(X_fused_test)[:, 1] / (5.0 * float(len(seeds)))

oof_voice_probs = np.zeros((N, len(voice_classes)))
test_voice_probs = np.zeros((len(test_raw), len(voice_classes)))
X_voice_sub = X_fused_train[voice_mask_all]
y_voice_sub = y_enc_voice

oof_video_probs = np.zeros((N, len(video_classes)))
test_video_probs = np.zeros((len(test_raw), len(video_classes)))
X_video_sub = X_fused_train[video_mask_all]
y_video_sub = y_enc_video

for seed in seeds:
    skf_v = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_v, val_v) in enumerate(skf_v.split(X_voice_sub, y_voice_sub)):
        orig_val_idx = np.where(voice_mask_all)[0][val_v]
        
        clf_lgb_v = LGBMClassifier(n_estimators=300, num_leaves=25, learning_rate=0.07, min_child_samples=10,
                                   colsample_bytree=0.75, subsample=0.85, random_state=seed+fold, verbosity=-1, n_jobs=-1)
        clf_xgb_v = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.80,
                                  colsample_bytree=0.75, tree_method='hist', random_state=seed+fold, n_jobs=-1)
        clf_cat_v = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.06, l2_leaf_reg=3,
                                       random_seed=seed+fold, verbose=0, thread_count=-1)
        clf_et_v = ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_split=3, random_state=seed+fold, n_jobs=-1)
        
        clf_lgb_v.fit(X_voice_sub[tr_v], y_voice_sub[tr_v])
        clf_xgb_v.fit(X_voice_sub[tr_v], y_voice_sub[tr_v])
        clf_cat_v.fit(X_voice_sub[tr_v], y_voice_sub[tr_v])
        clf_et_v.fit(X_voice_sub[tr_v], y_voice_sub[tr_v])
        
        p_val_v = (0.35 * clf_lgb_v.predict_proba(X_voice_sub[val_v]) + 
                   0.35 * clf_xgb_v.predict_proba(X_voice_sub[val_v]) + 
                   0.15 * clf_cat_v.predict_proba(X_voice_sub[val_v]) + 
                   0.15 * clf_et_v.predict_proba(X_voice_sub[val_v]))
        oof_voice_probs[orig_val_idx] += p_val_v / float(len(seeds))
        
        p_test_v = (0.35 * clf_lgb_v.predict_proba(X_fused_test) + 
                    0.35 * clf_xgb_v.predict_proba(X_fused_test) + 
                    0.15 * clf_cat_v.predict_proba(X_fused_test) + 
                    0.15 * clf_et_v.predict_proba(X_fused_test))
        test_voice_probs += p_test_v / (5.0 * float(len(seeds)))

    skf_vi = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr_vi, val_vi) in enumerate(skf_vi.split(X_video_sub, y_video_sub)):
        orig_val_idx_vi = np.where(video_mask_all)[0][val_vi]
        
        clf_lgb_vi = LGBMClassifier(n_estimators=300, num_leaves=25, learning_rate=0.07, min_child_samples=10,
                                    colsample_bytree=0.75, subsample=0.85, random_state=seed+fold, verbosity=-1, n_jobs=-1)
        clf_xgb_vi = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.80,
                                   colsample_bytree=0.75, tree_method='hist', random_state=seed+fold, n_jobs=-1)
        clf_cat_vi = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.06, l2_leaf_reg=3,
                                        random_seed=seed+fold, verbose=0, thread_count=-1)
        clf_et_vi = ExtraTreesClassifier(n_estimators=300, max_depth=18, min_samples_split=3, random_state=seed+fold, n_jobs=-1)
        
        clf_lgb_vi.fit(X_video_sub[tr_vi], y_video_sub[tr_vi])
        clf_xgb_vi.fit(X_video_sub[tr_vi], y_video_sub[tr_vi])
        clf_cat_vi.fit(X_video_sub[tr_vi], y_video_sub[tr_vi])
        clf_et_vi.fit(X_video_sub[tr_vi], y_video_sub[tr_vi])
        
        p_val_vi = (0.35 * clf_lgb_vi.predict_proba(X_video_sub[val_vi]) + 
                    0.35 * clf_xgb_vi.predict_proba(X_video_sub[val_vi]) + 
                    0.15 * clf_cat_vi.predict_proba(X_video_sub[val_vi]) + 
                    0.15 * clf_et_vi.predict_proba(X_video_sub[val_vi]))
        oof_video_probs[orig_val_idx_vi] += p_val_vi / float(len(seeds))
        
        p_test_vi = (0.35 * clf_lgb_vi.predict_proba(X_fused_test) + 
                     0.35 * clf_xgb_vi.predict_proba(X_fused_test) + 
                     0.15 * clf_cat_vi.predict_proba(X_fused_test) + 
                     0.15 * clf_et_vi.predict_proba(X_fused_test))
        test_video_probs += p_test_vi / (5.0 * float(len(seeds)))

oof_fused_10c = np.zeros((N, n_classes))
test_fused_10c = np.zeros((len(test_raw), n_classes))

for idx_v, c_v in enumerate(voice_classes):
    idx_10c = np.where(class_names == c_v)[0][0]
    oof_fused_10c[:, idx_10c] = (1.0 - oof_router_probs) * oof_voice_probs[:, idx_v]
    test_fused_10c[:, idx_10c] = (1.0 - test_router_probs) * test_voice_probs[:, idx_v]

for idx_vi, c_vi in enumerate(video_classes):
    idx_10c = np.where(class_names == c_vi)[0][0]
    oof_fused_10c[:, idx_10c] = oof_router_probs * oof_video_probs[:, idx_vi]
    test_fused_10c[:, idx_10c] = test_router_probs * test_video_probs[:, idx_vi]

oof_fused_10c = oof_fused_10c / (oof_fused_10c.sum(axis=1, keepdims=True) + 1e-9)
test_fused_10c = test_fused_10c / (test_fused_10c.sum(axis=1, keepdims=True) + 1e-9)

idx_zoom_v = np.where(class_names == "Zoom_video")[0][0]
idx_meet_v = np.where(class_names == "GoogleMeet_video")[0][0]
mask_zoom_meet = ((train_fe["label"] == "Zoom_video") | (train_fe["label"] == "GoogleMeet_video"))
X_zm = X_fused_train[mask_zoom_meet]
y_zm = (train_fe.loc[mask_zoom_meet, "label"] == "Zoom_video").astype(int).values

oof_zm_res = np.zeros(len(X_zm))
test_zm_res = np.zeros(len(test_raw))

skf_zm = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, val in skf_zm.split(X_zm, y_zm):
    clf_zm = LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                            min_child_samples=5, random_state=42, verbosity=-1, n_jobs=-1)
    clf_zm.fit(X_zm[tr], y_zm[tr])
    oof_zm_res[val] = clf_zm.predict_proba(X_zm[val])[:, 1]
    test_zm_res += clf_zm.predict_proba(X_fused_test)[:, 1] / 5.0

zm_orig_idx = np.where(mask_zoom_meet)[0]
oof_fused_10c[zm_orig_idx, idx_zoom_v] = 0.65 * oof_fused_10c[zm_orig_idx, idx_zoom_v] + 0.35 * oof_zm_res
oof_fused_10c[zm_orig_idx, idx_meet_v] = 0.65 * oof_fused_10c[zm_orig_idx, idx_meet_v] + 0.35 * (1.0 - oof_zm_res)

test_fused_10c[:, idx_zoom_v] = 0.65 * test_fused_10c[:, idx_zoom_v] + 0.35 * test_zm_res
test_fused_10c[:, idx_meet_v] = 0.65 * test_fused_10c[:, idx_meet_v] + 0.35 * (1.0 - test_zm_res)

idx_zoom_vc = np.where(class_names == "Zoom_voice")[0][0]
idx_disc_vc = np.where(class_names == "Discord_voice")[0][0]
mask_zoom_disc = ((train_fe["label"] == "Zoom_voice") | (train_fe["label"] == "Discord_voice"))
X_zd = X_fused_train[mask_zoom_disc]
y_zd = (train_fe.loc[mask_zoom_disc, "label"] == "Zoom_voice").astype(int).values

oof_zd_res = np.zeros(len(X_zd))
test_zd_res = np.zeros(len(test_raw))

skf_zd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, val in skf_zd.split(X_zd, y_zd):
    clf_zd = LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                            min_child_samples=5, random_state=42, verbosity=-1, n_jobs=-1)
    clf_zd.fit(X_zd[tr], y_zd[tr])
    oof_zd_res[val] = clf_zd.predict_proba(X_zd[val])[:, 1]
    test_zd_res += clf_zd.predict_proba(X_fused_test)[:, 1] / 5.0

zd_orig_idx = np.where(mask_zoom_disc)[0]
oof_fused_10c[zd_orig_idx, idx_zoom_vc] = 0.65 * oof_fused_10c[zd_orig_idx, idx_zoom_vc] + 0.35 * oof_zd_res
oof_fused_10c[zd_orig_idx, idx_disc_vc] = 0.65 * oof_fused_10c[zd_orig_idx, idx_disc_vc] + 0.35 * (1.0 - oof_zd_res)

test_fused_10c[:, idx_zoom_vc] = 0.65 * test_fused_10c[:, idx_zoom_vc] + 0.35 * test_zd_res
test_fused_10c[:, idx_disc_vc] = 0.65 * test_fused_10c[:, idx_disc_vc] + 0.35 * (1.0 - test_zd_res)

def f1_loss(weights):
    scaled_oof = oof_fused_10c * weights
    preds = np.argmax(scaled_oof, axis=1)
    return -f1_score(y_enc, preds, average="macro")

init_w = np.ones(n_classes)
opt_res = minimize(f1_loss, init_w, method="Nelder-Mead", options={"maxiter": 400})
best_weights = opt_res.x

calib_oof_probs = oof_fused_10c * best_weights
oof_preds = np.argmax(calib_oof_probs, axis=1)

macro_f1 = f1_score(y_enc, oof_preds, average="macro")
bal_acc = balanced_accuracy_score(y_enc, oof_preds)
overall_acc = accuracy_score(y_enc, oof_preds)

print(f"Macro-F1 Score: {macro_f1:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Overall Accuracy: {overall_acc*100:.2f}%")
print(classification_report(y_enc, oof_preds, target_names=class_names, digits=4))

cm = confusion_matrix(y_enc, oof_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"HMV-Stack Confusion Matrix (Macro-F1: {macro_f1:.4f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.close()

tta_test_probs = np.zeros((len(test_raw), n_classes))
time_idx = [0, 2, 4, 6, 8]
len_idx = [1, 3, 5, 7, 9]

for tta_pass in range(5):
    test_j_df = test_raw.copy()
    if tta_pass > 0:
        np.random.seed(tta_pass * 100)
        for k in range(1, 5):
            test_j_df[f"relative_time_{k}"] += np.random.uniform(-0.0001, 0.0001, size=len(test_raw))
            test_j_df[f"packet_length_{k}"] = np.clip(test_j_df[f"packet_length_{k}"] + np.random.choice([-1, 0, 1], size=len(test_raw)), 28, 1500)
    test_j_fe = extract_transport_features(test_j_df)
    test_j_seq = build_sequence_tensor(test_j_df)
    with torch.no_grad():
        test_j_emb, _ = gru_model(torch.tensor(test_j_seq, dtype=torch.float32).to(device))
        test_j_emb = test_j_emb.cpu().numpy()
    X_j_test = np.hstack([test_j_fe[selected_feats].values, test_j_emb])
    
    pass_probs = test_fused_10c * best_weights
    tta_test_probs += pass_probs / 5.0

final_test_preds_enc = np.argmax(tta_test_probs, axis=1)
final_test_labels = class_names[final_test_preds_enc]

submission_df = pd.DataFrame({"Predicted_Label": final_test_labels})
submission_df.to_csv("results/submission.csv", index=False)

prob_cols = [f"prob_{c}" for c in class_names]
prob_df = pd.DataFrame(tta_test_probs, columns=prob_cols)
prob_df["Predicted_Label"] = final_test_labels
prob_df.to_csv("results/predictions.csv", index=False)

test_with_preds = test_raw.copy()
test_with_preds["Predicted_Label"] = final_test_labels
test_with_preds.to_csv("results/Testing_set_with_predictions.csv", index=False)

print("Saved submission to results/submission.csv")
