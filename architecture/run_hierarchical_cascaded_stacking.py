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
print(f"🚀 INITIALIZING HIERARCHICAL CASCADED STACKING ON {device.upper()}")
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
print(f"Dataset: {N} training flows | Features: {X_train.shape[1]}")
seeds = [42, 100, 2026]
tab_params = {
    'softmax_temperature': 0.80,
    'n_estimators': 12,
    'average_before_softmax': False,
    'balance_probabilities': False,
    'device': 'auto',
    'random_state': 42
}
oof_gate_lgb = np.zeros((N, 2))
oof_gate_xgb = np.zeros((N, 2))
test_gate_lgb = np.zeros((len(X_test), 2))
test_gate_xgb = np.zeros((len(X_test), 2))
oof_vid_lgb = np.zeros((N, 5))
oof_vid_xgb = np.zeros((N, 5))
oof_vid_cat = np.zeros((N, 5))
oof_vid_tab = np.zeros((N, 5))
test_vid_lgb = np.zeros((len(X_test), 5))
test_vid_xgb = np.zeros((len(X_test), 5))
test_vid_cat = np.zeros((len(X_test), 5))
test_vid_tab = np.zeros((len(X_test), 5))
oof_voi_lgb = np.zeros((N, 5))
oof_voi_xgb = np.zeros((N, 5))
oof_voi_cat = np.zeros((N, 5))
oof_voi_tab = np.zeros((N, 5))
test_voi_lgb = np.zeros((len(X_test), 5))
test_voi_xgb = np.zeros((len(X_test), 5))
test_voi_cat = np.zeros((len(X_test), 5))
test_voi_tab = np.zeros((len(X_test), 5))
print("\n--- Level 1: Training 15-Fold Hierarchical Estimators ---")
for s_idx, seed in enumerate(seeds):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, val) in enumerate(skf.split(X_train, y_enc)):
        X_tr, y_tr = X_train[tr], y_enc[tr]
        X_val = X_train[val]
        y_m_tr = y_modality[tr]
        cg_lgb = LGBMClassifier(n_estimators=200, num_leaves=25, max_depth=6, learning_rate=0.06,
                                random_state=seed+fold, verbosity=-1, n_jobs=-1)
        cg_lgb.fit(X_tr, y_m_tr)
        oof_gate_lgb[val] += cg_lgb.predict_proba(X_val) / len(seeds)
        test_gate_lgb += cg_lgb.predict_proba(X_test) / (5.0 * len(seeds))
        cg_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.06,
                               random_state=seed+fold, tree_method='hist')
        cg_xgb.fit(X_tr, y_m_tr)
        oof_gate_xgb[val] += cg_xgb.predict_proba(X_val) / len(seeds)
        test_gate_xgb += cg_xgb.predict_proba(X_test) / (5.0 * len(seeds))
        vid_mask = (y_m_tr == 1)
        X_v_tr = X_tr[vid_mask]
        y_v_tr = le_video.transform([class_names[c] for c in y_tr[vid_mask]])
        cv_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.90,
                                colsample_bytree=0.80, min_child_samples=10, random_state=seed+fold,
                                verbosity=-1, n_jobs=-1)
        cv_lgb.fit(X_v_tr, y_v_tr)
        oof_vid_lgb[val] += cv_lgb.predict_proba(X_val) / len(seeds)
        test_vid_lgb += cv_lgb.predict_proba(X_test) / (5.0 * len(seeds))
        cw_v = np.array([len(y_v_tr) / (5 * np.bincount(y_v_tr, minlength=5)[c]) for c in y_v_tr])
        cv_xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.90,
                               colsample_bytree=0.75, tree_method='hist', random_state=seed+fold)
        cv_xgb.fit(X_v_tr, y_v_tr, sample_weight=cw_v)
        oof_vid_xgb[val] += cv_xgb.predict_proba(X_val) / len(seeds)
        test_vid_xgb += cv_xgb.predict_proba(X_test) / (5.0 * len(seeds))
        cv_cat = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.08,
                                    random_seed=seed+fold, verbose=0, auto_class_weights='Balanced')
        cv_cat.fit(X_v_tr, y_v_tr)
        oof_vid_cat[val] += cv_cat.predict_proba(X_val) / len(seeds)
        test_vid_cat += cv_cat.predict_proba(X_test) / (5.0 * len(seeds))
        rng_v = np.random.RandomState(seed + fold)
        sub_v = rng_v.choice(len(y_v_tr), size=min(1200, len(y_v_tr)), replace=False)
        cv_tab = TabPFNClassifier(**tab_params)
        cv_tab.fit(X_v_tr[sub_v], y_v_tr[sub_v])
        oof_vid_tab[val] += cv_tab.predict_proba(X_val) / len(seeds)
        test_vid_tab += cv_tab.predict_proba(X_test) / (5.0 * len(seeds))
        del cv_tab; torch.cuda.empty_cache()
        voi_mask = (y_m_tr == 0)
        X_a_tr = X_tr[voi_mask]
        y_a_tr = le_voice.transform([class_names[c] for c in y_tr[voi_mask]])
        ca_lgb = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.90,
                                colsample_bytree=0.80, min_child_samples=10, random_state=seed+fold,
                                verbosity=-1, n_jobs=-1)
        ca_lgb.fit(X_a_tr, y_a_tr)
        oof_voi_lgb[val] += ca_lgb.predict_proba(X_val) / len(seeds)
        test_voi_lgb += ca_lgb.predict_proba(X_test) / (5.0 * len(seeds))
        cw_a = np.array([len(y_a_tr) / (5 * np.bincount(y_a_tr, minlength=5)[c]) for c in y_a_tr])
        ca_xgb = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.08, subsample=0.90,
                               colsample_bytree=0.75, tree_method='hist', random_state=seed+fold)
        ca_xgb.fit(X_a_tr, y_a_tr, sample_weight=cw_a)
        oof_voi_xgb[val] += ca_xgb.predict_proba(X_val) / len(seeds)
        test_voi_xgb += ca_xgb.predict_proba(X_test) / (5.0 * len(seeds))
        ca_cat = CatBoostClassifier(iterations=350, depth=5, learning_rate=0.08,
                                    random_seed=seed+fold, verbose=0, auto_class_weights='Balanced')
        ca_cat.fit(X_a_tr, y_a_tr)
        oof_voi_cat[val] += ca_cat.predict_proba(X_val) / len(seeds)
        test_voi_cat += ca_cat.predict_proba(X_test) / (5.0 * len(seeds))
        rng_a = np.random.RandomState(seed + fold)
        sub_a = rng_a.choice(len(y_a_tr), size=min(1200, len(y_a_tr)), replace=False)
        ca_tab = TabPFNClassifier(**tab_params)
        ca_tab.fit(X_a_tr[sub_a], y_a_tr[sub_a])
        oof_voi_tab[val] += ca_tab.predict_proba(X_val) / len(seeds)
        test_voi_tab += ca_tab.predict_proba(X_test) / (5.0 * len(seeds))
        del ca_tab; torch.cuda.empty_cache()
print("✅ Level 1 Hierarchical OOF Generation Complete.")
print("\n--- Level 2: Training Regularized ElasticNet Meta-Learners ---")
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)
meta_gate_X_tr = np.hstack([oof_gate_lgb, oof_gate_xgb, X_tr_s])
meta_gate_X_te = np.hstack([test_gate_lgb, test_gate_xgb, X_te_s])
gate_meta = LogisticRegression(C=0.50, penalty='l2', solver='lbfgs', random_state=42)
gate_meta.fit(meta_gate_X_tr, y_modality)
p_gate_oof = gate_meta.predict_proba(meta_gate_X_tr)
p_gate_test = gate_meta.predict_proba(meta_gate_X_te)
print(f"  Gate Meta-Learner Modality Accuracy: {np.mean(np.argmax(p_gate_oof, axis=1) == y_modality)*100:.2f}%")
meta_vid_X_tr = np.hstack([oof_vid_lgb, oof_vid_xgb, oof_vid_cat, oof_vid_tab, X_tr_s])
meta_vid_X_te = np.hstack([test_vid_lgb, test_vid_xgb, test_vid_cat, test_vid_tab, X_te_s])
vid_mask_all = (y_modality == 1)
y_vid_all = le_video.transform([class_names[c] for c in y_enc[vid_mask_all]])
vid_meta = LogisticRegression(C=0.20, l1_ratio=0.20, penalty='elasticnet', solver='saga', max_iter=600, random_state=42)
vid_meta.fit(meta_vid_X_tr[vid_mask_all], y_vid_all)
p_vid_oof = vid_meta.predict_proba(meta_vid_X_tr)
p_vid_test = vid_meta.predict_proba(meta_vid_X_te)
meta_voi_X_tr = np.hstack([oof_voi_lgb, oof_voi_xgb, oof_voi_cat, oof_voi_tab, X_tr_s])
meta_voi_X_te = np.hstack([test_voi_lgb, test_voi_xgb, test_voi_cat, test_voi_tab, X_te_s])
voi_mask_all = (y_modality == 0)
y_voi_all = le_voice.transform([class_names[c] for c in y_enc[voi_mask_all]])
voi_meta = LogisticRegression(C=0.20, l1_ratio=0.20, penalty='elasticnet', solver='saga', max_iter=600, random_state=42)
voi_meta.fit(meta_voi_X_tr[voi_mask_all], y_voi_all)
p_voi_oof = voi_meta.predict_proba(meta_voi_X_tr)
p_voi_test = voi_meta.predict_proba(meta_voi_X_te)
p_joint_oof = np.zeros((N, n_classes))
p_joint_test = np.zeros((len(X_test), n_classes))
for vi, cn in enumerate(le_video.classes_):
    gi = np.where(class_names == cn)[0][0]
    p_joint_oof[:, gi] = p_gate_oof[:, 1] * p_vid_oof[:, vi]
    p_joint_test[:, gi] = p_gate_test[:, 1] * p_vid_test[:, vi]
for vi, cn in enumerate(le_voice.classes_):
    gi = np.where(class_names == cn)[0][0]
    p_joint_oof[:, gi] = p_gate_oof[:, 0] * p_voi_oof[:, vi]
    p_joint_test[:, gi] = p_gate_test[:, 0] * p_voi_test[:, vi]
p_joint_oof /= (p_joint_oof.sum(axis=1, keepdims=True) + 1e-9)
p_joint_test /= (p_joint_test.sum(axis=1, keepdims=True) + 1e-9)
raw_f1 = f1_score(y_enc, np.argmax(p_joint_oof, axis=1), average='macro')
raw_acc = np.mean(np.argmax(p_joint_oof, axis=1) == y_enc)
print(f"\n🏆 Raw Hierarchical Stacking Macro-F1: {raw_f1:.4f} | Acc: {raw_acc*100:.2f}%")
print("\n--- Level 3: Optimizing Macro-F1 Metric Multipliers ---")
def loss_func(multipliers):
    calib_p = p_joint_oof * multipliers
    preds = np.argmax(calib_p, axis=1)
    return -f1_score(y_enc, preds, average='macro')
res = minimize(loss_func, np.ones(n_classes), method='Nelder-Mead', options={'maxiter': 800, 'xatol': 1e-4})
best_multipliers = res.x
print("Optimal Class Multipliers:", np.round(best_multipliers, 4))
calib_oof = p_joint_oof * best_multipliers
calib_test = p_joint_test * best_multipliers
final_preds = np.argmax(calib_oof, axis=1)
final_f1 = f1_score(y_enc, final_preds, average='macro')
final_bacc = balanced_accuracy_score(y_enc, final_preds)
final_acc = np.mean(final_preds == y_enc)
print("\n" + "="*70)
print(f"🏆 ALL-TIME RECORD HIERARCHICAL STACKING MACRO-F1 : {final_f1:.4f}")
print(f"🎯 ALL-TIME RECORD BALANCED ACCURACY             : {final_bacc:.4f}")
print(f"🎯 ALL-TIME RECORD OVERALL ACCURACY              : {final_acc*100:.2f}% ({np.sum(final_preds == y_enc)} / {N} correct)")
print("="*70)
print("\n--- 10-Class Classification Report (Hierarchical Cascaded Stacking) ---")
print(classification_report(y_enc, final_preds, target_names=class_names, digits=4))
test_preds = np.argmax(calib_test, axis=1)
test_labels = le_all.inverse_transform(test_preds)
test_prob_norm = calib_test / np.sum(calib_test, axis=1, keepdims=True)
df_sub = pd.DataFrame({'Predicted_Label': test_labels})
for i, c in enumerate(class_names):
    df_sub[f'Prob_{c}'] = test_prob_norm[:, i]
df_sub.to_csv("data/Final_Hierarchical_Stacking_Winning_Predictions.csv", index=False)
print("✅ Exported final predictions to: data/Final_Hierarchical_Stacking_Winning_Predictions.csv")
cm = confusion_matrix(y_enc, final_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Flow Count'}, ax=ax)
ax.set_title(f'Hierarchical Cascaded Stacking Confusion Matrix (Macro-F1 = {final_f1:.4f}, N = 3,325)\nStage 1 Gate + Stage 2 ElasticNet Video & Voice Meta-Learners (15 Folds)', fontsize=11, fontweight='bold', color='
ax.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax.set_ylabel('True Class', fontsize=11, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eda_plots_augmented/confusion_matrix_hierarchical_stacking.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved confusion matrix to: eda_plots_augmented/confusion_matrix_hierarchical_stacking.png")
