import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import os
def engineer_features(df_in):
    df = df_in.copy()
    if 'relative_time_0' in df.columns:
        df = df.drop(columns=['relative_time_0'])
    times = [np.zeros(len(df))] + [df[f'relative_time_{i}'].values for i in range(1, 5)]
    lengths = [df[f'packet_length_{i}'].values for i in range(5)]
    for i in range(1, 5):
        df[f'delta_t_{i}'] = times[i] - times[i-1]
        df[f'delta_len_{i}'] = lengths[i] - lengths[i-1]
    for i in range(2, 5):
        df[f'accel_t_{i}'] = df[f'delta_t_{i}'] - df[f'delta_t_{i-1}']
        df[f'accel_len_{i}'] = df[f'delta_len_{i}'] - df[f'delta_len_{i-1}']
    for i in range(1, 5):
        df[f'rate_{i}'] = lengths[i] / (df[f'delta_t_{i}'] + 1e-9)
    df['overall_throughput'] = sum(lengths) / (times[4] + 1e-9)
    for i in range(1, 5):
        df[f'len_ratio_{i}_to_0'] = lengths[i] / (lengths[0] + 1e-9)
        df[f'len_ratio_{i}_to_{i-1}'] = lengths[i] / (lengths[i-1] + 1e-9)
    cum_len = lengths[0].copy()
    for i in range(1, 5):
        cum_len = cum_len + lengths[i]
        df[f'cumsum_len_{i}'] = cum_len
    len_matrix = np.column_stack(lengths)
    df['len_mean'] = np.mean(len_matrix, axis=1)
    df['len_std'] = np.std(len_matrix, axis=1)
    df['len_var'] = np.var(len_matrix, axis=1)
    df['len_min'] = np.min(len_matrix, axis=1)
    df['len_max'] = np.max(len_matrix, axis=1)
    df['len_range'] = df['len_max'] - df['len_min']
    df['len_skew'] = skew(len_matrix, axis=1)
    df['len_kurtosis'] = kurtosis(len_matrix, axis=1)
    dt_matrix = np.column_stack([df[f'delta_t_{i}'] for i in range(1, 5)])
    df['dt_mean'] = np.mean(dt_matrix, axis=1)
    df['dt_std'] = np.std(dt_matrix, axis=1)
    df['dt_min'] = np.min(dt_matrix, axis=1)
    df['dt_max'] = np.max(dt_matrix, axis=1)
    df['dt_range'] = df['dt_max'] - df['dt_min']
    df['dt_skew'] = skew(dt_matrix, axis=1)
    df['time_len_interaction'] = df['len_std'] * df['dt_std']
    return df
df_train = pd.read_csv("data/Training_set.csv")
df_train_eng = engineer_features(df_train)
features_all = [c for c in df_train_eng.columns if c != 'label']
X_raw = df_train_eng[features_all].copy()
y = df_train_eng['label']
vars = X_raw.var()
drop_vars = vars[vars == 0].index.tolist()
X_raw = X_raw.drop(columns=drop_vars)
corr = spearmanr(X_raw).statistic
corr_matrix = np.abs(corr)
upper = np.triu(corr_matrix, k=1)
to_drop = [X_raw.columns[i] for i in range(upper.shape[1]) if any(upper[:, i] > 0.90)]
X_pruned = X_raw.drop(columns=to_drop)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
clf.fit(X_pruned, y_encoded)
selector = SelectFromModel(clf, prefit=True, threshold="mean")
features_kept_idx = selector.get_support(indices=True)
final_features = X_pruned.columns[features_kept_idx].tolist()
df_train_final = pd.concat([X_pruned[final_features], y], axis=1)
df_train_final.to_csv("data/Training_set_selected.csv", index=False)
df_test = pd.read_csv("data/Testing_set.csv")
df_test_eng = engineer_features(df_test)
df_test_final = df_test_eng[final_features]
df_test_final.to_csv("data/Testing_set_selected.csv", index=False)
