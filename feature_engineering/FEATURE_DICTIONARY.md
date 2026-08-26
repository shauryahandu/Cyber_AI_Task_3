# Feature Engineering Dictionary: Encrypted RTC Traffic Invariants

This document details the complete feature space used by the **Hierarchical Multi-View Stacking (HMV-Stack)** model to classify encrypted 5-packet network flows.

---

## 1. Multi-View Feature Space Overview (118 Total Features)

The feature representation fuses two complementary views of the 5-packet network flow:
* **View 1 (54 Handcrafted Transport Invariants):** Domain-specific acoustic, wavelet, Markov, and physical timing features.
* **View 2 (64 Spatio-Temporal Sequence Features):** 2-layer Bidirectional GRU latent embeddings learned directly from the raw 4-channel tensor.

---

## 2. Handcrafted Domain Feature Categories (View 1)

### A. Codec-Specific Acoustic Pacing & Jitter Harmonics
* `opus_jitter_dev_1..4`: Absolute standardized deviation from expected 20ms WebRTC audio pacing: $|\Delta t_i - 0.020| / 0.020$.
* `mean_opus_jitter_dev`: Global acoustic jitter stability score.
* `min_opus_jitter_dev`: Minimum frame alignment error.

### B. Discrete Wavelet Energy Decomposition (Haar Wavelet)
* `haar_energy_ratio`: Ratio of high-frequency detail wavelet energy to low-frequency approximation energy across packet lengths:
  $$\frac{h_{d0}^2 + h_{d1}^2}{h_{a0}^2 + h_{a1}^2 + 10^{-9}}$$
* `haar_timing_ratio`: Detail-to-approximation energy ratio across inter-arrival times.

### C. First-Order Markov Transition Probabilities
* `p_voice_to_voice`: Probability of consecutive small voice packets ($100 \le L \le 350$ bytes).
* `p_large_to_large`: Probability of consecutive large video payload packets ($L > 800$ bytes).
* `p_small_to_large`: Probability of transition from silence/header to video frame clump.

### D. Statistical Payload Moments & Spectral Centroids
* `pl_mean`, `pl_std`, `pl_min`, `pl_max`, `pl_range`, `pl_median`, `pl_sum`: Basic dispersion moments.
* `pl_skew`, `pl_kurtosis`: Higher-order shape statistics.
* `pl_cv`: Coefficient of variation $\sigma_L / \mu_L$.
* `spectral_centroid`: Frequency-weighted packet energy center $\sum i \cdot L_i / \sum L_i$.
* `spectral_spread`: Second spectral moment measuring spectral variance.

### E. Information-Theoretic Entropy
* `packet_length_entropy`: Shannon entropy over 100-byte binned payload distribution:
  $$H(L) = -\sum p_i \log_2(p_i)$$

### F. Transport Dynamics & Autocorrelation
* `byte_rate`: Instantaneous flow throughput $\sum L_i / t_4$.
* `clump_to_silence_ratio`: Ratio of minimum burst interval to maximum silence interval.
* `autocorr_lag1`, `autocorr_lag2`: Serial dependency across consecutive packet sizes.
* `pl_gradient_slope`: Linear gradient slope $(L_4 - L_0) / 4$.

---

## 3. Spatio-Temporal Sequence Tensor & Bi-GRU Embeddings (View 2)

Input Tensor Shape: $(B, C_{\text{in}} = 4, T = 5)$:
1. **Channel 0 (Normalized Length):** $L_i / 1500.0$
2. **Channel 1 (Normalized IAT):** $\Delta t_i / (\max(\Delta t) + 10^{-5})$
3. **Channel 2 (Packet Acceleration):** $\Delta^2 t_i = \Delta t_i - \Delta t_{i-1}$
4. **Channel 3 (Cumulative Energy):** Running byte transfer sum $\sum_{j=0}^i L_j / \sum_{j=0}^4 L_j$

The tensor is processed through:
$$\text{Conv1D}(4 \to 32) \to \text{BatchNorm} \to \text{ReLU} \to \text{Bi-GRU}(32 \to 32) \to \text{Global Mean Pooling} \to 64\text{-D Representation}$$

---

## 4. Visualizations

* 📊 **Top 25 Feature Importance:** [`plots/top_25_features_importance.png`](plots/top_25_features_importance.png)
* 📈 **Packet Length Distributions:** [`plots/packet_length_distributions.png`](plots/packet_length_distributions.png)
* 🔬 **Throughput vs. Jitter Phase Space:** [`plots/iat_jitter_separation.png`](plots/iat_jitter_separation.png)
* 🎵 **Harmonic & Pacing Separation:** [`plots/harmonic_and_pacing_separation.png`](plots/harmonic_and_pacing_separation.png)
* 📉 **Entropy vs. Payload Variance:** [`plots/duty_cycle_and_entropy.png`](plots/duty_cycle_and_entropy.png)
