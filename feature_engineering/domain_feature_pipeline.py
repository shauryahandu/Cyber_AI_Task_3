import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
def extract_advanced_domain_features(df_in):
    df = df_in.copy()
    if 'relative_time_0' in df.columns:
        df = df.drop(columns=['relative_time_0'])
    times = [np.zeros(len(df))] + [df[f'relative_time_{i}'].values for i in range(1, 5)]
    lengths = [df[f'packet_length_{i}'].values for i in range(5)]
    len_matrix = np.column_stack(lengths)
    time_matrix = np.column_stack(times)
    dt_list = []
    dlen_list = []
    for i in range(1, 5):
        dt = times[i] - times[i-1]
        dlen = lengths[i] - lengths[i-1]
        df[f'delta_t_{i}'] = dt
        df[f'delta_len_{i}'] = dlen
        dt_list.append(dt)
        dlen_list.append(dlen)
    dt_matrix = np.column_stack(dt_list)
    dlen_matrix = np.column_stack(dlen_list)
    jitter_list = []
    for i in range(1, 4):
        jitter = np.abs(dt_list[i] - dt_list[i-1])
        accel_len = dlen_list[i] - dlen_list[i-1]
        df[f'jitter_{i+1}'] = jitter
        df[f'accel_len_{i+1}'] = accel_len
        jitter_list.append(jitter)
    jitter_matrix = np.column_stack(jitter_list)
    df['jitter_mean'] = np.mean(jitter_matrix, axis=1)
    df['jitter_std'] = np.std(jitter_matrix, axis=1)
    df['jitter_max'] = np.max(jitter_matrix, axis=1)
    df['jitter_cv'] = df['jitter_std'] / (df['jitter_mean'] + 1e-9)
    df['mtu_count'] = np.sum(len_matrix >= 1000, axis=1)
    df['mtu_ratio'] = df['mtu_count'] / 5.0
    df['small_pkt_count'] = np.sum(len_matrix <= 100, axis=1)
    df['small_pkt_ratio'] = df['small_pkt_count'] / 5.0
    df['mid_pkt_count'] = np.sum((len_matrix > 100) & (len_matrix < 1000), axis=1)
    df['is_all_voice_size'] = (np.sum(len_matrix <= 300, axis=1) == 5).astype(float)
    df['is_all_mtu'] = (np.sum(len_matrix >= 1000, axis=1) == 5).astype(float)
    norm_lens = len_matrix / (np.sum(len_matrix, axis=1, keepdims=True) + 1e-9)
    norm_lens_safe = np.clip(norm_lens, 1e-9, 1.0)
    df['len_entropy'] = -np.sum(norm_lens_safe * np.log2(norm_lens_safe), axis=1)
    fft_lens = np.abs(np.fft.rfft(len_matrix, axis=1))
    df['fft_len_dc'] = fft_lens[:, 0]
    df['fft_len_fund'] = fft_lens[:, 1]
    df['fft_len_harmonic'] = fft_lens[:, 2]
    fft_dts = np.abs(np.fft.rfft(dt_matrix, axis=1))
    df['fft_dt_dc'] = fft_dts[:, 0]
    df['fft_dt_fund'] = fft_dts[:, 1]
    for i in range(1, 5):
        df[f'rate_{i}'] = lengths[i] / (dt_list[i-1] + 1e-9)
    df['total_bytes'] = np.sum(len_matrix, axis=1)
    df['total_time'] = times[4]
    df['overall_throughput'] = df['total_bytes'] / (df['total_time'] + 1e-9)
    df['peak_rate'] = np.max(np.column_stack([df[f'rate_{i}'] for i in range(1, 5)]), axis=1)
    df['peak_to_avg_rate'] = df['peak_rate'] / (df['overall_throughput'] + 1e-9)
    for i in range(1, 5):
        df[f'len_ratio_{i}_to_0'] = lengths[i] / (lengths[0] + 1e-9)
        df[f'len_ratio_{i}_to_{i-1}'] = lengths[i] / (lengths[i-1] + 1e-9)
    df['len_max_to_min_ratio'] = np.max(len_matrix, axis=1) / (np.min(len_matrix, axis=1) + 1e-9)
    df['len_mean'] = np.mean(len_matrix, axis=1)
    df['len_std'] = np.std(len_matrix, axis=1)
    df['len_var'] = np.var(len_matrix, axis=1)
    df['len_min'] = np.min(len_matrix, axis=1)
    df['len_max'] = np.max(len_matrix, axis=1)
    df['len_median'] = np.median(len_matrix, axis=1)
    df['len_range'] = df['len_max'] - df['len_min']
    df['len_skew'] = skew(len_matrix, axis=1)
    df['len_kurtosis'] = kurtosis(len_matrix, axis=1)
    df['len_cv'] = df['len_std'] / (df['len_mean'] + 1e-9)
    df['len_iqr'] = np.percentile(len_matrix, 75, axis=1) - np.percentile(len_matrix, 25, axis=1)
    df['dt_mean'] = np.mean(dt_matrix, axis=1)
    df['dt_std'] = np.std(dt_matrix, axis=1)
    df['dt_min'] = np.min(dt_matrix, axis=1)
    df['dt_max'] = np.max(dt_matrix, axis=1)
    df['dt_median'] = np.median(dt_matrix, axis=1)
    df['dt_range'] = df['dt_max'] - df['dt_min']
    df['dt_skew'] = skew(dt_matrix, axis=1)
    df['dt_cv'] = df['dt_std'] / (df['dt_mean'] + 1e-9)
    df['time_len_interaction'] = df['len_std'] * df['dt_std']
    df['avg_bytes_per_hop'] = df['len_mean'] / (df['dt_mean'] + 1e-9)
    df['size_time_correlation'] = [np.corrcoef(len_matrix[row], time_matrix[row])[0, 1] for row in range(len(df))]
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df
df_train = pd.read_csv("data/Training_set.csv")
df_test = pd.read_csv("data/Testing_set.csv")
df_train_adv = extract_advanced_domain_features(df_train)
df_test_adv = extract_advanced_domain_features(df_test)
feature_cols = [c for c in df_train_adv.columns if c != 'label']
X_raw = df_train_adv[feature_cols].copy()
y = df_train_adv['label']
le = LabelEncoder()
y_enc = le.fit_transform(y)
vars_ = X_raw.var()
drop_zero = vars_[vars_ == 0].index.tolist()
X_raw = X_raw.drop(columns=drop_zero)
corr = spearmanr(X_raw).statistic
corr_matrix = np.abs(corr)
upper = np.triu(corr_matrix, k=1)
to_drop = [X_raw.columns[i] for i in range(upper.shape[1]) if any(upper[:, i] > 0.92)]
X_pruned = X_raw.drop(columns=to_drop)
clf = ExtraTreesClassifier(n_estimators=300, random_state=42)
clf.fit(X_pruned, y_enc)
selector = SelectFromModel(clf, prefit=True, threshold="0.4*mean")
kept_idx = selector.get_support(indices=True)
final_cols = X_pruned.columns[kept_idx].tolist()
df_train_out = pd.concat([X_pruned[final_cols], y], axis=1)
df_train_out.to_csv("data/Training_set_domain.csv", index=False)
df_test_out = df_test_adv[final_cols]
df_test_out.to_csv("data/Testing_set_domain.csv", index=False)
