import time
import warnings
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from features import extract_transport_features, build_sequence_tensor
from model import BiGRUEmbedder

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

train_raw = pd.read_csv("data/Augmented_Training_set.csv")
test_raw = pd.read_csv("data/Testing_set.csv")

train_fe = extract_transport_features(train_raw)
test_fe = extract_transport_features(test_raw)

le_all = LabelEncoder()
y_enc = le_all.fit_transform(train_fe["label"])
class_names = le_all.classes_
n_classes = len(class_names)
N = len(train_raw)

y_modality = np.array([1 if "video" in c else 0 for c in train_fe["label"]])
voice_mask_all = (y_modality == 0)
video_mask_all = (y_modality == 1)

le_voice = LabelEncoder()
y_enc_voice = le_voice.fit_transform(train_fe.loc[voice_mask_all, "label"])
voice_classes = le_voice.classes_

le_video = LabelEncoder()
y_enc_video = le_video.fit_transform(train_fe.loc[video_mask_all, "label"])
video_classes = le_video.classes_

raw_feats = [c for c in train_fe.columns if c not in ["label", "relative_time_0"]]
corr_matrix = pd.DataFrame(train_fe[raw_feats].values, columns=raw_feats).corr(method='spearman').abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [c for c in upper_tri.columns if any(upper_tri[c] > 0.98)]
non_coll_feats = [c for c in raw_feats if c not in drop_cols]

mi_scores = mutual_info_classif(train_fe[non_coll_feats].values, y_enc, random_state=42)
clf_lgb_s = LGBMClassifier(n_estimators=250, num_leaves=31, learning_rate=0.08, random_state=42, verbosity=-1, n_jobs=-1)
clf_lgb_s.fit(train_fe[non_coll_feats].values, y_enc)
imp_comb = (0.5 * (mi_scores / mi_scores.max()) + 0.5 * (clf_lgb_s.feature_importances_ / clf_lgb_s.feature_importances_.max()))
selected_feats = pd.Series(imp_comb, index=non_coll_feats).sort_values(ascending=False).head(54).index.tolist()

X_seq_train = build_sequence_tensor(train_raw)
X_seq_test = build_sequence_tensor(test_raw)

gru_model = BiGRUEmbedder().to(device)
opt_gru = optim.AdamW(gru_model.parameters(), lr=2e-3, weight_decay=1e-4)
crit_gru = nn.CrossEntropyLoss()
loader_gru = DataLoader(TensorDataset(torch.tensor(X_seq_train), torch.tensor(y_enc, dtype=torch.long)), batch_size=128, shuffle=True)

gru_model.train()
for epoch in range(1, 31):
 for bx, by in loader_gru:
 bx, by = bx.to(device), by.to(device)
 opt_gru.zero_grad()
 _, logits = gru_model(bx)
 loss = crit_gru(logits, by)
 loss.backward()
 opt_gru.step()

gru_model.eval()
with torch.no_grad():
 train_seq_emb, _ = gru_model(torch.tensor(X_seq_train).to(device))
 test_seq_emb, _ = gru_model(torch.tensor(X_seq_test).to(device))
 train_seq_emb = train_seq_emb.cpu().numpy()
 test_seq_emb = test_seq_emb.cpu().numpy()

X_fused_train = np.hstack([train_fe[selected_feats].values, train_seq_emb])
X_fused_test = np.hstack([test_fe[selected_feats].values, test_seq_emb])

oof_modality = np.zeros(N)
test_modality = np.zeros(len(test_raw))

skf_m = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr, val) in enumerate(skf_m.split(X_fused_train, y_modality)):
 clf_m = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.07, min_child_samples=10,
 colsample_bytree=0.75, subsample=0.85, random_state=42+fold, verbosity=-1, n_jobs=-1)
 clf_m.fit(X_fused_train[tr], y_modality[tr])
 oof_modality[val] = clf_m.predict_proba(X_fused_train[val])[:, 1]
 test_modality += clf_m.predict_proba(X_fused_test)[:, 1] / 5.0

mod_acc = accuracy_score(y_modality, (oof_modality >= 0.50).astype(int))

seeds = [42, 100, 2026]
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

for i, c_name in enumerate(class_names):
 if "voice" in c_name:
 v_idx = np.where(voice_classes == c_name)[0][0]
 oof_fused_10c[:, i] = (1.0 - oof_modality) * oof_voice_probs[:, v_idx]
 test_fused_10c[:, i] = (1.0 - test_modality) * test_voice_probs[:, v_idx]
 else:
 vi_idx = np.where(video_classes == c_name)[0][0]
 oof_fused_10c[:, i] = oof_modality * oof_video_probs[:, vi_idx]
 test_fused_10c[:, i] = test_modality * test_video_probs[:, vi_idx]

c_zvid = np.where(class_names == 'Zoom_video')[0][0]
c_mvid = np.where(class_names == 'GoogleMeet_video')[0][0]
mask_vid = np.isin(y_enc, [c_zvid, c_mvid])
clf_r_vid = LGBMClassifier(n_estimators=300, num_leaves=18, max_depth=5, learning_rate=0.04, random_state=42, verbosity=-1, n_jobs=-1)
clf_r_vid.fit(X_fused_train[mask_vid], (y_enc[mask_vid] == c_zvid).astype(int))

c_zvoi = np.where(class_names == 'Zoom_voice')[0][0]
c_dvoi = np.where(class_names == 'Discord_voice')[0][0]
mask_dvoi = np.isin(y_enc, [c_zvoi, c_dvoi])
clf_r_dvoi = LGBMClassifier(n_estimators=300, num_leaves=18, max_depth=5, learning_rate=0.04, random_state=42, verbosity=-1, n_jobs=-1)
clf_r_dvoi.fit(X_fused_train[mask_dvoi], (y_enc[mask_dvoi] == c_zvoi).astype(int))

def apply_resolvers(probs, feats):
 probs_adj = probs.copy()
 for i in range(len(probs)):
 top2 = np.argsort(probs[i])[-2:]
 margin = probs[i, top2[1]] - probs[i, top2[0]]
 if margin < 0.14:
 if set(top2) == {c_zvid, c_mvid}:
 p_res = clf_r_vid.predict_proba(feats[[i]])[0, 1]
 p_sum = probs[i, c_zvid] + probs[i, c_mvid]
 probs_adj[i, c_zvid] = p_sum * p_res
 probs_adj[i, c_mvid] = p_sum * (1.0 - p_res)
 elif set(top2) == {c_zvoi, c_dvoi}:
 p_res = clf_r_dvoi.predict_proba(feats[[i]])[0, 1]
 p_sum = probs[i, c_zvoi] + probs[i, c_dvoi]
 probs_adj[i, c_zvoi] = p_sum * p_res
 probs_adj[i, c_dvoi] = p_sum * (1.0 - p_res)
 return probs_adj

oof_resolved = apply_resolvers(oof_fused_10c, X_fused_train)

def loss_func(multipliers):
 calib_p = oof_resolved * multipliers
 preds = np.argmax(calib_p, axis=1)
 return -f1_score(y_enc, preds, average='macro')

res = minimize(loss_func, np.ones(n_classes), method='Nelder-Mead', options={'maxiter': 800, 'xatol': 1e-4})
best_multipliers = res.x

calib_oof = oof_resolved * best_multipliers
final_preds = np.argmax(calib_oof, axis=1)
final_f1 = f1_score(y_enc, final_preds, average='macro')
final_bacc = balanced_accuracy_score(y_enc, final_preds)
final_acc = np.mean(final_preds == y_enc)

print(f"Macro-F1 Score: {final_f1:.4f}")
print(f"Balanced Accuracy: {final_bacc:.4f}")
print(f"Overall Accuracy: {final_acc*100:.2f}%")
print(classification_report(y_enc, final_preds, target_names=class_names, digits=4))

tta_test_probs = np.zeros((len(test_raw), n_classes))
np.random.seed(42)

for tta_pass in range(5):
 if tta_pass == 0:
 test_j_df = test_raw.copy()
 else:
 test_j_df = test_raw.copy()
 for i in range(1, 5):
 jitter = np.random.normal(0, 2.5e-5, size=len(test_raw))
 test_j_df[f"relative_time_{i}"] = np.maximum(0, test_j_df[f"relative_time_{i}"] + jitter)

 test_j_fe = extract_transport_features(test_j_df)
 test_j_seq = build_sequence_tensor(test_j_df)
 with torch.no_grad():
 test_j_emb, _ = gru_model(torch.tensor(test_j_seq).to(device))
 test_j_emb = test_j_emb.cpu().numpy()
 
 X_test_j_fused = np.hstack([test_j_fe[selected_feats].values, test_j_emb])
 p_meta_j_res = apply_resolvers(test_fused_10c, X_test_j_fused)
 tta_test_probs += (p_meta_j_res * best_multipliers) / 5.0

test_preds = np.argmax(tta_test_probs, axis=1)
test_labels = le_all.inverse_transform(test_preds)
test_prob_norm = tta_test_probs / np.sum(tta_test_probs, axis=1, keepdims=True)

df_sub = pd.DataFrame({'Predicted_Label': test_labels})
for i, c in enumerate(class_names):
 df_sub[f'Prob_{c}'] = test_prob_norm[:, i]

df_sub.to_csv("results/predictions.csv", index=False)

cm = confusion_matrix(y_enc, final_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
 xticklabels=class_names, yticklabels=class_names,
 linewidths=0.5, linecolor='white', ax=ax)
ax.set_title(f'Hierarchical Multi-View Stacking Confusion Matrix (Macro-F1 = {final_f1:.4f}, N = {N})', fontsize=11)
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('True Class', fontsize=11)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
