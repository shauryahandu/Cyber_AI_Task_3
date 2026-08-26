import pandas as pd
import numpy as np
import warnings
import torch
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 INITIALIZING LEAN 3-MODEL CASCADED SUPER-PIPELINE ON {device.upper()}")
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pl = [f"packet_length_{i}" for i in range(5)]
    times = [np.zeros(len(df))] + [df[f'relative_time_{i}'].values for i in range(1, 5)]
    dt_list, dlen_list = [], []
    for i in range(1, 5):
        dt = df[f"relative_time_{i}"] - df[f"relative_time_{i-1}"]
        dlen = df[f"packet_length_{i}"] - df[f"packet_length_{i-1}"]
        df[f"iat_{i}"] = dt
        df[f"pl_delta_{i}"] = dlen
        dt_list.append(dt.values)
        dlen_list.append(dlen.values)
    dt_matrix = np.column_stack(dt_list)
    len_matrix = np.column_stack([df[f'packet_length_{i}'].values for i in range(5)])
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
    return df
train_raw = pd.read_csv("data/Augmented_Training_set.csv")
test_raw = pd.read_csv("data/Testing_set.csv")
train_df = engineer_features(train_raw)
test_df = engineer_features(test_raw)
le_all = LabelEncoder()
y_enc = le_all.fit_transform(train_df["label"])
class_names = le_all.classes_
n_classes = len(class_names)
is_video = np.array(['video' in c for c in class_names])
y_modality = is_video[y_enc].astype(int)
video_classes = [c for c in class_names if 'video' in c]
voice_classes = [c for c in class_names if 'voice' in c]
le_video = LabelEncoder().fit(video_classes)
le_voice = LabelEncoder().fit(voice_classes)
X_train_df = train_df.drop(columns=["label", "relative_time_0"], errors='ignore')
X_test_df = test_df[X_train_df.columns]
X_train = X_train_df.values
X_test = X_test_df.values
N = len(X_train)
print(f"Dataset Size: {N} flows | Features: {X_train.shape[1]}")
seeds = [42, 100, 2026]
tab_params = {
    'softmax_temperature': 0.80,
    'n_estimators': 12,
    'average_before_softmax': False,
    'balance_probabilities': False,
    'device': 'auto',
    'random_state': 42
}
def run_cascaded_evaluation(X_tr_data, y_tr_data, y_mod_tr_data, X_t_data, sample_weights=None):
    oof_probs = np.zeros((len(X_tr_data), n_classes))
    test_probs = np.zeros((len(X_t_data), n_classes))
    for s_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_seed = np.zeros((len(X_tr_data), n_classes))
        test_seed = np.zeros((len(X_t_data), n_classes))
        for fold, (tr, val) in enumerate(skf.split(X_tr_data, y_tr_data)):
            X_tr, y_tr = X_tr_data[tr], y_tr_data[tr]
            X_val, y_val = X_tr_data[val], y_tr_data[val]
            y_m_tr = y_mod_tr_data[tr]
            sw_tr = sample_weights[tr] if sample_weights is not None else None
            clf_g_lgb = LGBMClassifier(n_estimators=200, num_leaves=25, max_depth=6, learning_rate=0.06,
                                       random_state=seed+fold, verbosity=-1, n_jobs=-1)
            clf_g_lgb.fit(X_tr, y_m_tr, sample_weight=sw_tr)
            clf_g_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.06,
                                       random_state=seed+fold, tree_method='hist')
            clf_g_xgb.fit(X_tr, y_m_tr, sample_weight=sw_tr)
            p_mod_val = 0.60 * clf_g_lgb.predict_proba(X_val) + 0.40 * clf_g_xgb.predict_proba(X_val)
            p_mod_test = 0.60 * clf_g_lgb.predict_proba(X_t_data) + 0.40 * clf_g_xgb.predict_proba(X_t_data)
            vid_mask = (y_m_tr == 1)
            X_vid_tr = X_tr[vid_mask]
            y_vid_tr = le_video.transform([class_names[c] for c in y_tr[vid_mask]])
            sw_vid = sw_tr[vid_mask] if sw_tr is not None else None
            clf_v_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.90,
                                       colsample_bytree=0.80, min_child_samples=10, random_state=seed+fold,
                                       verbosity=-1, n_jobs=-1)
            clf_v_lgb.fit(X_vid_tr, y_vid_tr, sample_weight=sw_vid)
            p_vid_val_lgb = clf_v_lgb.predict_proba(X_val)
            p_vid_t_lgb = clf_v_lgb.predict_proba(X_t_data)
            cw_vid = np.array([len(y_vid_tr) / (5 * np.bincount(y_vid_tr, minlength=5)[c]) for c in y_vid_tr])
            final_sw_vid = cw_vid * (sw_vid if sw_vid is not None else 1.0)
            clf_v_xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.90,
                                       colsample_bytree=0.75, tree_method='hist', random_state=seed+fold)
            clf_v_xgb.fit(X_vid_tr, y_vid_tr, sample_weight=final_sw_vid)
            p_vid_val_xgb = clf_v_xgb.predict_proba(X_val)
            p_vid_t_xgb = clf_v_xgb.predict_proba(X_t_data)
            rng_v = np.random.RandomState(seed + fold)
            sub_v = rng_v.choice(len(y_vid_tr), size=min(1200, len(y_vid_tr)), replace=False)
            clf_v_tab = TabPFNClassifier(**tab_params)
            clf_v_tab.fit(X_vid_tr[sub_v], y_vid_tr[sub_v])
            p_vid_val_tab = clf_v_tab.predict_proba(X_val)
            p_vid_t_tab = clf_v_tab.predict_proba(X_t_data)
            del clf_v_tab; torch.cuda.empty_cache()
            p_vid_val = 0.55 * p_vid_val_lgb + 0.30 * p_vid_val_tab + 0.15 * p_vid_val_xgb
            p_vid_test = 0.55 * p_vid_t_lgb + 0.30 * p_vid_t_tab + 0.15 * p_vid_t_xgb
            voi_mask = (y_m_tr == 0)
            X_voi_tr = X_tr[voi_mask]
            y_voi_tr = le_voice.transform([class_names[c] for c in y_tr[voi_mask]])
            sw_voi = sw_tr[voi_mask] if sw_tr is not None else None
            clf_a_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.90,
                                       colsample_bytree=0.80, min_child_samples=10, random_state=seed+fold,
                                       verbosity=-1, n_jobs=-1)
            clf_a_lgb.fit(X_voi_tr, y_voi_tr, sample_weight=sw_voi)
            p_voi_val_lgb = clf_a_lgb.predict_proba(X_val)
            p_voi_t_lgb = clf_a_lgb.predict_proba(X_t_data)
            cw_voi = np.array([len(y_voi_tr) / (5 * np.bincount(y_voi_tr, minlength=5)[c]) for c in y_voi_tr])
            final_sw_voi = cw_voi * (sw_voi if sw_voi is not None else 1.0)
            clf_a_xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.90,
                                       colsample_bytree=0.75, tree_method='hist', random_state=seed+fold)
            clf_a_xgb.fit(X_voi_tr, y_voi_tr, sample_weight=final_sw_voi)
            p_voi_val_xgb = clf_a_xgb.predict_proba(X_val)
            p_voi_t_xgb = clf_a_xgb.predict_proba(X_t_data)
            rng_a = np.random.RandomState(seed + fold)
            sub_a = rng_a.choice(len(y_voi_tr), size=min(1200, len(y_voi_tr)), replace=False)
            clf_a_tab = TabPFNClassifier(**tab_params)
            clf_a_tab.fit(X_voi_tr[sub_a], y_voi_tr[sub_a])
            p_voi_val_tab = clf_a_tab.predict_proba(X_val)
            p_voi_t_tab = clf_a_tab.predict_proba(X_t_data)
            del clf_a_tab; torch.cuda.empty_cache()
            p_voi_val = 0.60 * p_voi_val_lgb + 0.30 * p_voi_val_xgb + 0.10 * p_voi_val_tab
            p_voi_test = 0.60 * p_voi_t_lgb + 0.30 * p_voi_t_xgb + 0.10 * p_voi_t_tab
            p_joint_val = np.zeros((len(X_val), n_classes))
            p_joint_test = np.zeros((len(X_t_data), n_classes))
            for vi, cn in enumerate(le_video.classes_):
                gi = np.where(class_names == cn)[0][0]
                p_joint_val[:, gi] = p_mod_val[:, 1] * p_vid_val[:, vi]
                p_joint_test[:, gi] = p_mod_test[:, 1] * p_vid_test[:, vi]
            for vi, cn in enumerate(le_voice.classes_):
                gi = np.where(class_names == cn)[0][0]
                p_joint_val[:, gi] = p_mod_val[:, 0] * p_voi_val[:, vi]
                p_joint_test[:, gi] = p_mod_test[:, 0] * p_voi_test[:, vi]
            p_joint_val /= (p_joint_val.sum(axis=1, keepdims=True) + 1e-9)
            p_joint_test /= (p_joint_test.sum(axis=1, keepdims=True) + 1e-9)
            oof_seed[val] = p_joint_val
            test_seed += p_joint_test / 5.0
        oof_probs += oof_seed / len(seeds)
        test_probs += test_seed / len(seeds)
    return oof_probs, test_probs
print("\n--- Phase 1: 15-Fold Multi-Seed Cascaded Evaluation ---")
base_oof_probs, base_test_probs = run_cascaded_evaluation(X_train, y_enc, y_modality, X_test)
base_preds = np.argmax(base_oof_probs, axis=1)
base_f1 = f1_score(y_enc, base_preds, average='macro')
base_acc = np.mean(base_preds == y_enc)
print(f"🏆 Base 15-Fold Macro-F1 (Before Transduction): {base_f1:.4f} | Acc: {base_acc*100:.2f}%")
print("\n--- Phase 2: Transductive High-Confidence Pseudo-Labeling ---")
test_max_p = np.max(base_test_probs, axis=1)
test_pseudo_y = np.argmax(base_test_probs, axis=1)
conf_mask = (test_max_p >= 0.93)
n_pseudo = np.sum(conf_mask)
print(f"Identified {n_pseudo} / {len(X_test)} high-confidence test flows (P >= 0.93, {n_pseudo/len(X_test)*100:.1f}%).")
if n_pseudo >= 30:
    X_pseudo = X_test[conf_mask]
    y_pseudo = test_pseudo_y[conf_mask]
    y_mod_pseudo = is_video[y_pseudo].astype(int)
    X_aug = np.vstack([X_train, X_pseudo])
    y_aug = np.concatenate([y_enc, y_pseudo])
    y_mod_aug = np.concatenate([y_modality, y_mod_pseudo])
    sw_aug = np.concatenate([np.ones(len(X_train)), 0.50 * np.ones(len(X_pseudo))])
    print(f"Enriched training manifold: {len(X_train)} original + {n_pseudo} pseudo-labeled = {len(X_aug)} total flows.")
    print("\n--- Phase 3: Retraining 15-Fold Cascaded Ensemble with Transduction ---")
    trans_oof_probs_full, trans_test_probs = run_cascaded_evaluation(X_aug, y_aug, y_mod_aug, X_test, sample_weights=sw_aug)
    final_oof_probs = trans_oof_probs_full[:len(X_train)]
    final_test_probs = trans_test_probs
else:
    print("Insufficient high-confidence flows; using base predictions.")
    final_oof_probs = base_oof_probs
    final_test_probs = base_test_probs
print("\n--- Phase 4: Macro-F1 Metric Multiplier Calibration ---")
def loss_func(multipliers):
    calib_p = final_oof_probs * multipliers
    preds = np.argmax(calib_p, axis=1)
    return -f1_score(y_enc, preds, average='macro')
res = minimize(loss_func, np.ones(n_classes), method='Nelder-Mead', options={'maxiter': 800, 'xatol': 1e-4})
best_multipliers = res.x
print("Optimal Class Multipliers:", np.round(best_multipliers, 4))
calib_oof = final_oof_probs * best_multipliers
calib_test = final_test_probs * best_multipliers
final_preds = np.argmax(calib_oof, axis=1)
final_f1 = f1_score(y_enc, final_preds, average='macro')
final_bacc = balanced_accuracy_score(y_enc, final_preds)
final_acc = np.mean(final_preds == y_enc)
print("\n" + "="*70)
print(f"🏆 NEW ALL-TIME PROJECT PEAK MACRO-F1 : {final_f1:.4f}")
print(f"🎯 NEW ALL-TIME PROJECT BALANCED ACC  : {final_bacc:.4f}")
print(f"🎯 NEW ALL-TIME PROJECT EXACT ACCURACY: {final_acc*100:.2f}% ({np.sum(final_preds == y_enc)} / {N} correct)")
print("="*70)
print("\n--- 10-Class Classification Report (Lean 3-Model Cascaded Super-Pipeline) ---")
print(classification_report(y_enc, final_preds, target_names=class_names, digits=4))
test_preds = np.argmax(calib_test, axis=1)
test_labels = le_all.inverse_transform(test_preds)
test_prob_norm = calib_test / np.sum(calib_test, axis=1, keepdims=True)
df_sub = pd.DataFrame({'Predicted_Label': test_labels})
for i, c in enumerate(class_names):
    df_sub[f'Prob_{c}'] = test_prob_norm[:, i]
df_sub.to_csv("data/Final_Lean_3Model_Winning_Predictions.csv", index=False)
print("✅ Exported winning predictions to: data/Final_Lean_3Model_Winning_Predictions.csv")
cm = confusion_matrix(y_enc, final_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Flow Count'}, ax=ax)
ax.set_title(f'Lean 3-Model Cascaded Super-Pipeline Confusion Matrix (Macro-F1 = {final_f1:.4f}, N = 3,325)\n15-Fold Multi-Seed (LightGBM + XGBoost + TabPFN v3) + Transduction', fontsize=12, fontweight='bold', color='
ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots_augmented/confusion_matrix_lean_3model.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved confusion matrix to: eda_plots_augmented/confusion_matrix_lean_3model.png")
