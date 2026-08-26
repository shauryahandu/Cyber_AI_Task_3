import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy

def extract_transport_features(df: pd.DataFrame) -> pd.DataFrame:
 df = df.copy()
 pl = [f"packet_length_{i}" for i in range(5)]
 times = [f"relative_time_{i}" for i in range(5)]

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
 df["pl_skew"] = df[pl].skew(axis=1).fillna(0)
 df["pl_kurtosis"] = df[pl].apply(lambda r: float(kurtosis(r)), axis=1).fillna(0)
 df["pl_cv"] = df["pl_std"] / (df["pl_mean"] + 1e-9)

 iat_cols = [f"iat_{i}" for i in range(1, 5)]
 df["iat_mean"] = df[iat_cols].mean(axis=1)
 df["iat_std"] = df[iat_cols].std(axis=1)
 df["iat_max"] = df[iat_cols].max(axis=1)
 df["iat_min"] = df[iat_cols].min(axis=1)
 df["iat_sum"] = df["relative_time_4"]
 df["iat_skew"] = df[iat_cols].apply(lambda r: float(skew(r)), axis=1).fillna(0)
 df["iat_kurtosis"] = df[iat_cols].apply(lambda r: float(kurtosis(r)), axis=1).fillna(0)
 df["iat_relative_variance"] = (df["iat_std"]**2) / (df["iat_mean"] + 1e-6)

 for i in range(1, 5):
 df[f"opus_jitter_dev_{i}"] = np.abs(df[f"iat_{i}"] - 0.020) / 0.020
 df["mean_opus_jitter_dev"] = df[[f"opus_jitter_dev_{i}" for i in range(1, 5)]].mean(axis=1)
 df["min_opus_jitter_dev"] = df[[f"opus_jitter_dev_{i}" for i in range(1, 5)]].min(axis=1)

 def calc_entropy(row):
 clipped = np.clip((row.values // 100).astype(int), 0, 15)
 counts = np.bincount(clipped, minlength=16)
 probs = counts[counts > 0] / float(len(row))
 return entropy(probs, base=2)
 df["packet_length_entropy"] = df[pl].apply(calc_entropy, axis=1)

 pl_arr = df[pl].values
 freq_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
 df["spectral_centroid"] = np.sum(pl_arr * freq_weights, axis=1) / (df["pl_sum"] + 1e-9)
 df["spectral_spread"] = np.sqrt(np.sum(pl_arr * ((freq_weights - df["spectral_centroid"].values[:, None])**2), axis=1) / (df["pl_sum"] + 1e-9))

 p_small_to_large = np.zeros(len(df))
 p_large_to_large = np.zeros(len(df))
 p_voice_to_voice = np.zeros(len(df))
 for k in range(len(df)):
 trans_s_l, trans_l_l, trans_v_v = 0, 0, 0
 for step in range(4):
 if pl_arr[k, step] < 200 and pl_arr[k, step+1] > 800:
 trans_s_l += 1
 if pl_arr[k, step] > 800 and pl_arr[k, step+1] > 800:
 trans_l_l += 1
 if (100 <= pl_arr[k, step] <= 350) and (100 <= pl_arr[k, step+1] <= 350):
 trans_v_v += 1
 p_small_to_large[k] = trans_s_l / 4.0
 p_large_to_large[k] = trans_l_l / 4.0
 p_voice_to_voice[k] = trans_v_v / 4.0
 df["p_small_to_large"] = p_small_to_large
 df["p_large_to_large"] = p_large_to_large
 df["p_voice_to_voice"] = p_voice_to_voice

 r1_list, r2_list = [], []
 for k in range(len(df)):
 s = pl_arr[k]
 s_diff = s - np.mean(s)
 var = np.sum(s_diff**2) + 1e-9
 r1_list.append(np.sum(s_diff[:4] * s_diff[1:]) / var)
 r2_list.append(np.sum(s_diff[:3] * s_diff[2:]) / var)
 df["autocorr_lag1"] = r1_list
 df["autocorr_lag2"] = r2_list
 df["pl_gradient_slope"] = (df["packet_length_4"] - df["packet_length_0"]) / 4.0
 df["pl_time_gradient"] = (df["packet_length_4"] - df["packet_length_0"]) / (df["relative_time_4"] + 1e-6)

 df["byte_rate"] = df["pl_sum"] / (df["relative_time_4"] + 1e-6)
 rest_cols = [f"packet_length_{i}" for i in range(1, 5)]
 df["pl0_to_mean_rest"] = df["packet_length_0"] / (df[rest_cols].mean(axis=1) + 1e-9)
 df["n_small_pkts"] = (df[pl] < 100).sum(axis=1)
 df["n_voice_pkts"] = ((df[pl] >= 100) & (df[pl] <= 350)).sum(axis=1)
 df["n_video_pkts"] = (df[pl] > 900).sum(axis=1)
 df["n_fast_iat"] = (df[iat_cols] < 0.001).sum(axis=1)

 for c in iat_cols + ["iat_sum", "iat_mean"]:
 df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
 for c in pl + ["pl_mean", "pl_std", "byte_rate"]:
 df[f"log_{c}"] = np.log1p(df[c])

 dt_min_safe = np.minimum.reduce(dt_list)
 dt_min_safe = np.where(dt_min_safe < 1e-7, 1e-7, dt_min_safe)
 dt_max_safe = np.maximum.reduce(dt_list)
 df["clump_to_silence_ratio"] = dt_min_safe / (dt_max_safe + 1e-6)
 df["pacing_payload_interaction"] = np.log1p(df["iat_mean"]) * np.log1p(df["pl_std"])

 pld0, pld1 = df["pl_delta_1"].values, df["pl_delta_2"].values
 pld2, pld3 = df["pl_delta_3"].values, df["pl_delta_4"].values
 h_a0 = (pld0 + pld1) / np.sqrt(2)
 h_a1 = (pld2 + pld3) / np.sqrt(2)
 h_d0 = (pld0 - pld1) / np.sqrt(2)
 h_d1 = (pld2 - pld3) / np.sqrt(2)
 df["haar_energy_ratio"] = (h_d0**2 + h_d1**2) / (h_a0**2 + h_a1**2 + 1e-9)

 dt0, dt1 = df["iat_1"].values, df["iat_2"].values
 dt2, dt3 = df["iat_3"].values, df["iat_4"].values
 dt_a0 = (dt0 + dt1) / np.sqrt(2)
 dt_a1 = (dt2 + dt3) / np.sqrt(2)
 dt_d0 = (dt0 - dt1) / np.sqrt(2)
 dt_d1 = (dt2 - dt3) / np.sqrt(2)
 df["haar_timing_ratio"] = (dt_d0**2 + dt_d1**2) / (dt_a0**2 + dt_a1**2 + 1e-9)

 return df

def build_sequence_tensor(df: pd.DataFrame) -> np.ndarray:
 pl = [f"packet_length_{i}" for i in range(5)]
 times = [f"relative_time_{i}" for i in range(5)]
 lens = df[pl].values / 1500.0
 ts = df[times].values
 iats = np.zeros((len(df), 5))
 iats[:, 1:] = np.diff(ts, axis=1) / (np.max(np.diff(ts, axis=1), axis=1, keepdims=True) + 1e-5)
 accel = np.zeros((len(df), 5))
 accel[:, 2:] = np.diff(iats[:, 1:], axis=1)
 cum_energy = np.cumsum(df[pl].values, axis=1) / (np.sum(df[pl].values, axis=1, keepdims=True) + 1e-6)
 tensor = np.stack([lens, iats, accel, cum_energy], axis=1)
 return tensor.astype(np.float32)
