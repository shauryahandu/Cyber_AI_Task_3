import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def extract_transport_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pl = [f"packet_length_{i}" for i in range(5)]
    times = [f"relative_time_{i}" for i in range(5)]

    pl_arr = df[pl].values.astype(np.float64)
    ts_arr = df[times].values.astype(np.float64)
    iat_arr = np.diff(ts_arr, axis=1)
    pld_arr = np.diff(pl_arr, axis=1)

    for i in range(1, 5):
        df[f"iat_{i}"] = iat_arr[:, i-1]
        df[f"pl_delta_{i}"] = pld_arr[:, i-1]

    df["pl_mean"] = np.mean(pl_arr, axis=1)
    df["pl_std"] = np.std(pl_arr, axis=1)
    df["pl_min"] = np.min(pl_arr, axis=1)
    df["pl_max"] = np.max(pl_arr, axis=1)
    df["pl_range"] = df["pl_max"] - df["pl_min"]
    df["pl_median"] = np.median(pl_arr, axis=1)
    df["pl_sum"] = np.sum(pl_arr, axis=1)
    df["pl_skew"] = df[pl].skew(axis=1).fillna(0)
    df["pl_kurtosis"] = np.nan_to_num(kurtosis(pl_arr, axis=1), nan=0.0)
    df["pl_cv"] = df["pl_std"] / (df["pl_mean"] + 1e-9)

    iat_cols = [f"iat_{i}" for i in range(1, 5)]
    df["iat_mean"] = np.mean(iat_arr, axis=1)
    df["iat_std"] = np.std(iat_arr, axis=1)
    df["iat_max"] = np.max(iat_arr, axis=1)
    df["iat_min"] = np.min(iat_arr, axis=1)
    df["iat_sum"] = df["relative_time_4"]
    df["iat_skew"] = np.nan_to_num(skew(iat_arr, axis=1), nan=0.0)
    df["iat_kurtosis"] = np.nan_to_num(kurtosis(iat_arr, axis=1), nan=0.0)
    df["iat_relative_variance"] = (df["iat_std"]**2) / (df["iat_mean"] + 1e-6)

    for i in range(1, 5):
        df[f"opus_jitter_dev_{i}"] = np.abs(df[f"iat_{i}"] - 0.020) / 0.020
    df["mean_opus_jitter_dev"] = df[[f"opus_jitter_dev_{i}" for i in range(1, 5)]].mean(axis=1)
    df["min_opus_jitter_dev"] = df[[f"opus_jitter_dev_{i}" for i in range(1, 5)]].min(axis=1)

    clipped_bins = np.clip((pl_arr // 100).astype(int), 0, 15)
    one_hot = np.eye(16)[clipped_bins]
    hist = one_hot.sum(axis=1) / 5.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -np.sum(np.where(hist > 0, hist * np.log2(hist), 0), axis=1)
    df["packet_length_entropy"] = ent

    freq_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    df["spectral_centroid"] = np.sum(pl_arr * freq_weights, axis=1) / (df["pl_sum"] + 1e-9)
    spread_diff = freq_weights[None, :] - df["spectral_centroid"].values[:, None]
    df["spectral_spread"] = np.sqrt(np.sum(pl_arr * (spread_diff**2), axis=1) / (df["pl_sum"] + 1e-9))

    is_s0 = (pl_arr[:, :-1] < 200)
    is_l1 = (pl_arr[:, 1:] > 800)
    df["p_small_to_large"] = np.mean(is_s0 & is_l1, axis=1)

    is_l0 = (pl_arr[:, :-1] > 800)
    df["p_large_to_large"] = np.mean(is_l0 & is_l1, axis=1)

    is_v0 = (pl_arr[:, :-1] >= 100) & (pl_arr[:, :-1] <= 350)
    is_v1 = (pl_arr[:, 1:] >= 100) & (pl_arr[:, 1:] <= 350)
    df["p_voice_to_voice"] = np.mean(is_v0 & is_v1, axis=1)

    s_diff = pl_arr - df["pl_mean"].values[:, None]
    var = np.sum(s_diff**2, axis=1) + 1e-9
    df["autocorr_lag1"] = np.sum(s_diff[:, :4] * s_diff[:, 1:], axis=1) / var
    df["autocorr_lag2"] = np.sum(s_diff[:, :3] * s_diff[:, 2:], axis=1) / var

    t_diff = iat_arr - df["iat_mean"].values[:, None]
    t_var = np.sum(t_diff**2, axis=1) + 1e-9
    df["iat_autocorr_lag1"] = np.sum(t_diff[:, :3] * t_diff[:, 1:], axis=1) / t_var
    df["iat_autocorr_lag2"] = np.sum(t_diff[:, :2] * t_diff[:, 2:], axis=1) / t_var

    iat_accel = np.diff(iat_arr, n=2, axis=1)
    df["iat_accel_0"] = iat_accel[:, 0]
    df["iat_accel_1"] = iat_accel[:, 1]
    df["iat_curvature_ratio"] = np.mean(np.abs(iat_accel), axis=1) / (df["iat_mean"] + 1e-6)

    df["pl_gradient_slope"] = (df["packet_length_4"] - df["packet_length_0"]) / 4.0
    df["pl_time_gradient"] = (df["packet_length_4"] - df["packet_length_0"]) / (df["relative_time_4"] + 1e-6)

    df["byte_rate"] = df["pl_sum"] / (df["relative_time_4"] + 1e-6)
    rest_cols = [f"packet_length_{i}" for i in range(1, 5)]
    df["pl0_to_mean_rest"] = df["packet_length_0"] / (df[rest_cols].mean(axis=1) + 1e-9)
    df["n_small_pkts"] = (pl_arr < 100).sum(axis=1)
    df["n_voice_pkts"] = ((pl_arr >= 100) & (pl_arr <= 350)).sum(axis=1)
    df["n_video_pkts"] = (pl_arr > 900).sum(axis=1)
    df["n_fast_iat"] = (iat_arr < 0.001).sum(axis=1)

    df["n_opus_dtx"] = ((pl_arr >= 30) & (pl_arr <= 80)).sum(axis=1)

    df["pl_mod4_aligned"] = ((pl_arr.astype(int) % 4) == 0).sum(axis=1) / 5.0
    df["pl_mod8_aligned"] = ((pl_arr.astype(int) % 8) == 0).sum(axis=1) / 5.0
    df["pl_mod16_aligned"] = ((pl_arr.astype(int) % 16) == 0).sum(axis=1) / 5.0
    df["pl_mod32_aligned"] = ((pl_arr.astype(int) % 32) == 0).sum(axis=1) / 5.0

    df["pl_zoom_audio_envelope"] = (((pl_arr >= 106) & (pl_arr <= 118)) | ((pl_arr >= 78) & (pl_arr <= 86))).sum(axis=1) / 5.0
    df["pl_meet_audio_envelope"] = (((pl_arr >= 210) & (pl_arr <= 255)) | ((pl_arr >= 140) & (pl_arr <= 175))).sum(axis=1) / 5.0
    df["pl_video_slice_ratio"] = (pl_arr > 1150).sum(axis=1) / 5.0
    df["iat_sub_ms_burst_ratio"] = (iat_arr < 0.0006).sum(axis=1) / 4.0

    opus_nodes = np.array([58.0, 64.0, 76.0, 88.0, 96.0, 108.0])
    dists = np.min(np.abs(pl_arr[:, :, None] - opus_nodes[None, None, :]), axis=2)
    df["opus_vbr_residual_min"] = np.min(dists, axis=1)
    df["opus_vbr_residual_mean"] = np.mean(dists, axis=1)

    burst_rates = pl_arr[:, 1:] / (iat_arr + 1e-6)
    df["burst_rate_mean"] = np.mean(burst_rates, axis=1)
    df["burst_rate_max"] = np.max(burst_rates, axis=1)
    df["burst_rate_std"] = np.std(burst_rates, axis=1)
    burst_accel = np.diff(burst_rates, axis=1) / (iat_arr[:, 1:] + 1e-6)
    df["burst_accel_std"] = np.std(burst_accel, axis=1)
    df["max_burst_energy_ratio"] = np.max(pl_arr, axis=1) / (df["pl_sum"] + 1e-9)

    df["pld_abs_mean"] = np.mean(np.abs(pld_arr), axis=1)
    df["pld_std"] = np.std(pld_arr, axis=1)
    df["pld_skew"] = np.nan_to_num(skew(pld_arr, axis=1), nan=0.0)
    df["pld_kurtosis"] = np.nan_to_num(kurtosis(pld_arr, axis=1), nan=0.0)
    df["webrtc_pacing_residual"] = np.mean(np.abs(iat_arr - np.median(iat_arr, axis=1, keepdims=True)), axis=1)
    df["webrtc_audio_pacing_20ms"] = (np.abs(iat_arr - 0.020) < 0.002).sum(axis=1) / 4.0
    df["zoom_sscr_marker"] = ((pl_arr[:, 0] == 110) | (pl_arr[:, 0] == 82)).astype(float)
    df["meet_stun_keepalive"] = ((pl_arr < 120).sum(axis=1) >= 2).astype(float)

    for c in iat_cols + ["iat_sum", "iat_mean"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    for c in pl + ["pl_mean", "pl_std", "byte_rate"]:
        df[f"log_{c}"] = np.log1p(df[c])

    dt_min = np.min(iat_arr, axis=1)
    dt_max = np.max(iat_arr, axis=1)
    df["clump_to_silence_ratio"] = np.maximum(dt_min, 1e-7) / (dt_max + 1e-6)
    df["pacing_payload_interaction"] = np.log1p(df["iat_mean"]) * np.log1p(df["pl_std"])

    pld0, pld1 = pld_arr[:, 0], pld_arr[:, 1]
    pld2, pld3 = pld_arr[:, 2], pld_arr[:, 3]
    h_a0 = (pld0 + pld1) / np.sqrt(2)
    h_a1 = (pld2 + pld3) / np.sqrt(2)
    h_d0 = (pld0 - pld1) / np.sqrt(2)
    h_d1 = (pld2 - pld3) / np.sqrt(2)
    df["haar_energy_ratio"] = (h_d0**2 + h_d1**2) / (h_a0**2 + h_a1**2 + 1e-9)

    dt0, dt1 = iat_arr[:, 0], iat_arr[:, 1]
    dt2, dt3 = iat_arr[:, 2], iat_arr[:, 3]
    dt_a0 = (dt0 + dt1) / np.sqrt(2)
    dt_a1 = (dt2 + dt3) / np.sqrt(2)
    dt_d0 = (dt0 - dt1) / np.sqrt(2)
    dt_d1 = (dt2 - dt3) / np.sqrt(2)
    df["haar_timing_ratio"] = (dt_d0**2 + dt_d1**2) / (dt_a0**2 + dt_a1**2 + 1e-9)

    return df

def build_sequence_tensor(df: pd.DataFrame) -> np.ndarray:
    pl = [f"packet_length_{i}" for i in range(5)]
    times = [f"relative_time_{i}" for i in range(5)]
    lens = df[pl].values.astype(np.float32) / 1500.0
    ts = df[times].values.astype(np.float32)
    iats = np.zeros((len(df), 5), dtype=np.float32)
    diffs = np.diff(ts, axis=1)
    iats[:, 1:] = diffs / (np.max(diffs, axis=1, keepdims=True) + 1e-5)
    accel = np.zeros((len(df), 5), dtype=np.float32)
    accel[:, 2:] = np.diff(iats[:, 1:], axis=1)
    cum_energy = np.cumsum(df[pl].values.astype(np.float32), axis=1) / (np.sum(df[pl].values.astype(np.float32), axis=1, keepdims=True) + 1e-6)
    tensor = np.stack([lens, iats, accel, cum_energy], axis=1)
    return tensor
