import pandas as pd
import numpy as np
import warnings
import os
import torch
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pl = [f"packet_length_{i}" for i in range(5)]
    dt_list, dlen_list = [], []
    for i in range(1, 5):
        dt = df[f"relative_time_{i}"] - df[f"relative_time_{i-1}"]
        dlen = df[f"packet_length_{i}"] - df[f"packet_length_{i-1}"]
        df[f"iat_{i}"] = dt
        df[f"pl_delta_{i}"] = dlen
        dt_list.append(dt.values)
        dlen_list.append(dlen.values)
    df["pl_mean"] = df[pl].mean(axis=1)
    df["pl_std"] = df[pl].std(axis=1)
    df["pl_min"] = df[pl].min(axis=1)
    df["pl_max"] = df[pl].max(axis=1)
    df["pl_range"] = df["pl_max"] - df["pl_min"]
    df["pl_median"] = df[pl].median(axis=1)
    df["pl_sum"] = df[pl].sum(axis=1)
    df["pl_skew"] = df[pl].skew(axis=1)
    df["pl_cv"] = df["pl_std"] / (df["pl_mean"] + 1e-9)
    iat_cols = [f"iat_{i}" for i in range(1, 5)]
    df["iat_mean"] = df[iat_cols].mean(axis=1)
    df["iat_std"] = df[iat_cols].std(axis=1)
    df["iat_max"] = df[iat_cols].max(axis=1)
    df["iat_min"] = df[iat_cols].min(axis=1)
    df["iat_sum"] = df["relative_time_4"]
    for c in iat_cols + ["iat_sum"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    for c in pl:
        df[f"log_{c}"] = np.log1p(df[c])
    df["n_small_pkts"] = (df[pl] < 100).sum(axis=1)
    df["n_large_pkts"] = (df[pl] > 1200).sum(axis=1)
    rest_cols = [f"packet_length_{i}" for i in range(1, 5)]
    df["pl0_to_mean_rest"] = df["packet_length_0"] / (df[rest_cols].mean(axis=1) + 1e-9)
    df["n_fast_iat"] = (df[iat_cols] < 0.001).sum(axis=1)
    df["byte_rate"] = df["pl_sum"] / (df["relative_time_4"] + 1e-6)
    df["log_iat_pl_ratio"] = np.log1p(df["iat_mean"]) / (np.log1p(df["pl_mean"]) + 1e-6)
    df["log_pl_cv"] = np.log1p(df["pl_cv"])
    df["log_vad_silence"] = np.log1p(df["n_small_pkts"])
    df["log_fast_ratio"] = df["n_fast_iat"] / 4.0
    dt_min_safe = np.minimum.reduce(dt_list)
    dt_min_safe = np.where(dt_min_safe < 1e-7, 1e-7, dt_min_safe)
    dt_max_safe = np.maximum.reduce(dt_list)
    df["clump_to_silence_ratio"] = dt_min_safe / (dt_max_safe + 1e-6)
    df["pacing_payload_interaction"] = np.log1p(df["iat_mean"]) * np.log1p(df["pl_std"])
    pld0 = df["pl_delta_1"].values
    pld1 = df["pl_delta_2"].values
    pld2 = df["pl_delta_3"].values
    pld3 = df["pl_delta_4"].values
    h_a0 = (pld0 + pld1) / np.sqrt(2)
    h_a1 = (pld2 + pld3) / np.sqrt(2)
    h_d0 = (pld0 - pld1) / np.sqrt(2)
    h_d1 = (pld2 - pld3) / np.sqrt(2)
    df["haar_energy_ratio"] = (h_d0**2 + h_d1**2) / (h_a0**2 + h_a1**2 + 1e-9)
    dt0 = df["iat_1"].values
    dt1 = df["iat_2"].values
    dt2 = df["iat_3"].values
    dt3 = df["iat_4"].values
    dt_a0 = (dt0 + dt1) / np.sqrt(2)
    dt_a1 = (dt2 + dt3) / np.sqrt(2)
    dt_d0 = (dt0 - dt1) / np.sqrt(2)
    dt_d1 = (dt2 - dt3) / np.sqrt(2)
    df["haar_timing_ratio"] = (dt_d0**2 + dt_d1**2) / (dt_a0**2 + dt_a1**2 + 1e-9)
    df["voice_active_count"] = ((df[pl] >= 100) & (df[pl] <= 350)).sum(axis=1)
    df["vad_silence_ratio"] = df["n_small_pkts"] / 5.0
    return df
train_raw = pd.read_csv("data/Augmented_Training_set.csv")
test_raw = pd.read_csv("data/Testing_set.csv")
train_df = engineer_features(train_raw)
test_df = engineer_features(test_raw)
le_all = LabelEncoder()
y_enc = le_all.fit_transform(train_df["label"])
class_names = le_all.classes_
n_classes = len(class_names)
X_train_df = train_df.drop(columns=["label", "relative_time_0"], errors='ignore')
X_test_df = test_df[X_train_df.columns]
X_train = X_train_df.values
X_test = X_test_df.values
N = len(X_train)
seeds = [42, 100, 2026]
oof_lgb = np.zeros((N, n_classes))
oof_xgb = np.zeros((N, n_classes))
oof_tab = np.zeros((N, n_classes))
oof_cat = np.zeros((N, n_classes))
test_lgb = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_tab = np.zeros((len(X_test), n_classes))
test_cat = np.zeros((len(X_test), n_classes))
tab_params = {
    'softmax_temperature': 0.80,
    'n_estimators': 12,
    'average_before_softmax': False,
    'balance_probabilities': False,
    'device': 'auto',
    'random_state': 42
}
for s_idx, seed in enumerate(seeds):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, val) in enumerate(skf.split(X_train, y_enc)):
        X_tr, y_tr = X_train[tr], y_enc[tr]
        X_val = X_train[val]
        clf_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.90,
                                 colsample_bytree=0.80, min_child_samples=10, random_state=seed+fold,
                                 verbosity=-1, n_jobs=-1)
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val] += clf_lgb.predict_proba(X_val) / len(seeds)
        test_lgb += clf_lgb.predict_proba(X_test) / (5.0 * len(seeds))
        sw_tr = np.array([len(y_tr) / (n_classes * np.bincount(y_tr, minlength=n_classes)[c]) for c in y_tr])
        clf_xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.90,
                                colsample_bytree=0.75, tree_method='hist', random_state=seed+fold)
        clf_xgb.fit(X_tr, y_tr, sample_weight=sw_tr)
        oof_xgb[val] += clf_xgb.predict_proba(X_val) / len(seeds)
        test_xgb += clf_xgb.predict_proba(X_test) / (5.0 * len(seeds))
        clf_cat = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.08,
                                     random_seed=seed+fold, verbose=0, auto_class_weights='Balanced')
        clf_cat.fit(X_tr, y_tr)
        oof_cat[val] += clf_cat.predict_proba(X_val) / len(seeds)
        test_cat += clf_cat.predict_proba(X_test) / (5.0 * len(seeds))
        rng = np.random.RandomState(seed + fold)
        sub_idx = rng.choice(len(y_tr), size=min(1200, len(y_tr)), replace=False)
        clf_tab = TabPFNClassifier(**tab_params)
        clf_tab.fit(X_tr[sub_idx], y_tr[sub_idx])
        oof_tab[val] += clf_tab.predict_proba(X_val) / len(seeds)
        test_tab += clf_tab.predict_proba(X_test) / (5.0 * len(seeds))
        del clf_tab; torch.cuda.empty_cache()
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
meta_X_train = np.hstack([oof_lgb, oof_xgb, oof_cat, oof_tab, X_tr_scaled])
meta_X_test = np.hstack([test_lgb, test_xgb, test_cat, test_tab, X_test_scaled])
skf_l2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_meta = np.zeros((N, n_classes))
test_meta = np.zeros((len(X_test), n_classes))
for fold_idx, (tr_idx, val_idx) in enumerate(skf_l2.split(meta_X_train, y_enc)):
    clf_meta = LogisticRegression(C=0.20, l1_ratio=0.20, solver='saga', penalty='elasticnet', max_iter=600, random_state=42)
    clf_meta.fit(meta_X_train[tr_idx], y_enc[tr_idx])
    oof_meta[val_idx] = clf_meta.predict_proba(meta_X_train[val_idx])
    test_meta += clf_meta.predict_proba(meta_X_test) / 5.0
raw_meta_f1 = f1_score(y_enc, np.argmax(oof_meta, axis=1), average='macro')
c_zvid = np.where(class_names == 'Zoom_video')[0][0]
c_mvid = np.where(class_names == 'GoogleMeet_video')[0][0]
mask_vid_pair = np.isin(y_enc, [c_zvid, c_mvid])
y_vid_binary = (y_enc[mask_vid_pair] == c_zvid).astype(int)
clf_res_video = LGBMClassifier(n_estimators=250, num_leaves=15, max_depth=4, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.75, random_state=42, verbosity=-1, n_jobs=-1)
clf_res_video.fit(X_train[mask_vid_pair], y_vid_binary)
c_zvoi = np.where(class_names == 'Zoom_voice')[0][0]
c_mvoi = np.where(class_names == 'GoogleMeet_voice')[0][0]
mask_voi_pair = np.isin(y_enc, [c_zvoi, c_mvoi])
y_voi_binary = (y_enc[mask_voi_pair] == c_zvoi).astype(int)
clf_res_voice = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.85,
                              colsample_bytree=0.75, tree_method='hist', random_state=42)
clf_res_voice.fit(X_train[mask_voi_pair], y_voi_binary)
oof_resolved = oof_meta.copy()
n_vid_disambiguated = 0
n_voi_disambiguated = 0
for i in range(N):
    top2 = np.argsort(oof_meta[i])[-2:]
    margin = oof_meta[i, top2[1]] - oof_meta[i, top2[0]]
    if set(top2) == {c_zvid, c_mvid} and margin < 0.15:
        p_zvid_res = clf_res_video.predict_proba(X_train[[i]])[0, 1]
        p_sum_orig = oof_meta[i, c_zvid] + oof_meta[i, c_mvid]
        oof_resolved[i, c_zvid] = p_sum_orig * p_zvid_res
        oof_resolved[i, c_mvid] = p_sum_orig * (1.0 - p_zvid_res)
        n_vid_disambiguated += 1
    if set(top2) == {c_zvoi, c_mvoi} and margin < 0.15:
        p_zvoi_res = clf_res_voice.predict_proba(X_train[[i]])[0, 1]
        p_sum_orig = oof_meta[i, c_zvoi] + oof_meta[i, c_mvoi]
        oof_resolved[i, c_zvoi] = p_sum_orig * p_zvoi_res
        oof_resolved[i, c_mvoi] = p_sum_orig * (1.0 - p_zvoi_res)
        n_voi_disambiguated += 1
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
tta_test_probs = np.zeros((len(X_test), n_classes))
np.random.seed(42)
for tta_pass in range(5):
    if tta_pass == 0:
        test_jittered_df = test_raw.copy()
    else:
        test_jittered_df = test_raw.copy()
        for i in range(1, 5):
            jitter = np.random.normal(0, 2.5e-5, size=len(test_raw))
            test_jittered_df[f"relative_time_{i}"] = np.maximum(0, test_jittered_df[f"relative_time_{i}"] + jitter)
    test_j_fe = engineer_features(test_jittered_df)
    X_test_j = test_j_fe[X_train_df.columns].values
    X_test_j_s = scaler.transform(X_test_j)
    meta_X_test_j = np.hstack([test_lgb, test_xgb, test_cat, test_tab, X_test_j_s])
    p_meta_j = np.zeros((len(X_test), n_classes))
    for fold_idx, (tr_idx, val_idx) in enumerate(skf_l2.split(meta_X_train, y_enc)):
        clf_meta = LogisticRegression(C=0.20, l1_ratio=0.20, solver='saga', penalty='elasticnet', max_iter=600, random_state=42)
        clf_meta.fit(meta_X_train[tr_idx], y_enc[tr_idx])
        p_meta_j += clf_meta.predict_proba(meta_X_test_j) / 5.0
    for i in range(len(X_test)):
        top2 = np.argsort(p_meta_j[i])[-2:]
        margin = p_meta_j[i, top2[1]] - p_meta_j[i, top2[0]]
        if set(top2) == {c_zvid, c_mvid} and margin < 0.15:
            p_zvid_res = clf_res_video.predict_proba(X_test_j[[i]])[0, 1]
            p_sum_orig = p_meta_j[i, c_zvid] + p_meta_j[i, c_mvid]
            p_meta_j[i, c_zvid] = p_sum_orig * p_zvid_res
            p_meta_j[i, c_mvid] = p_sum_orig * (1.0 - p_zvid_res)
        if set(top2) == {c_zvoi, c_mvoi} and margin < 0.15:
            p_zvoi_res = clf_res_voice.predict_proba(X_test_j[[i]])[0, 1]
            p_sum_orig = p_meta_j[i, c_zvoi] + p_meta_j[i, c_mvoi]
            p_meta_j[i, c_zvoi] = p_sum_orig * p_zvoi_res
            p_meta_j[i, c_mvoi] = p_sum_orig * (1.0 - p_zvoi_res)
    tta_test_probs += (p_meta_j * best_multipliers) / 5.0
test_preds = np.argmax(tta_test_probs, axis=1)
test_labels = le_all.inverse_transform(test_preds)
test_prob_norm = tta_test_probs / np.sum(tta_test_probs, axis=1, keepdims=True)
df_sub = pd.DataFrame({'Predicted_Label': test_labels})
for i, c in enumerate(class_names):
    df_sub[f'Prob_{c}'] = test_prob_norm[:, i]
df_sub.to_csv("data/Final_Ensemble_Stacking_Resolvers_Final_Predictions.csv", index=False)
cm = confusion_matrix(y_enc, final_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Flow Count'}, ax=ax)
ax.set_title(f'Ensemble Stacking with Resolvers Confusion Matrix (Macro-F1 = {final_f1:.4f}, N = 3,325)\nLevel-1 (15 Folds) + Level-2 ElasticNet (96 Feats) + Pairwise Resolvers + TTA', fontsize=11, fontweight='bold', color='
ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots_augmented/confusion_matrix_ensemble_stacking_resolvers.png', dpi=300, bbox_inches='tight')
plt.close()
