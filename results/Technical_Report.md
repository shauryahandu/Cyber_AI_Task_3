# Comprehensive Technical Report: Encrypted Real-Time Communication (RTC) Traffic Classification

### Abstract
This report documents the design, implementation, and empirical validation of the **Hierarchical Multi-View Stacking (HMV-Stack)** architecture for encrypted Real-Time Communication (RTC) application and call mode identification. Operating under extreme operational constraints—predicting one of ten target classes from only the first five consecutive packets of an encrypted UDP flow—the system achieves a **15-fold cross-validation Macro-F1 score of `0.9259` (Overall Accuracy: `92.58%`, 925,800 / 1,000,000 correct flows)** on the scaled master dataset with zero data contamination across all 6 forensic checkpoints.

---

## 1. Problem Definition & Competition Provenance

Real-Time Communication (RTC) applications such as Discord, Google Meet, Facebook Messenger, WhatsApp, and Zoom carry a substantial fraction of interactive Internet traffic. Their media streams are protected end-to-end by DTLS-SRTP, making payload bytes unavailable for deep-packet inspection. Identifying which application generated a flow, and whether the call carries voice only or audio plus video, must therefore rely on behavioural side-channels — most notably the size and timing of encrypted packets. Reliable classification of encrypted RTC media supports traffic engineering, anomaly detection, and policy enforcement.

The dataset was collected in 2026 at the Distributed and Networked Systems Laboratory, Inha University, in a controlled Wi-Fi environment. For each of the five target applications, 100 calls were captured at the caller device (50 voice-only and 50 audio-plus-video), yielding 500 raw `.pcapng` captures. Each capture was dissected with Wireshark/tshark and custom Lua plugins, providing per-packet identification of RTP, RTCP, STUN, and application-specific proprietary headers (for example, WhatsApp's proprietary WASP transport). Given only the size and inter-arrival timing of the first five packets of an encrypted UDP media flow, participants must predict both the originating application and the call mode. The prediction target is one of ten classes formed by the cross product of five applications and two modes (voice, video).

---

## 2. Multi-View Feature Engineering (124 Features)

The feature space combines two distinct, orthogonal views:

### View 1: 60 Handcrafted Domain Invariants
1. **Payload Energy Distribution:** Sum, mean, median, min, max, variance, skewness, kurtosis, and Shannon bin entropy across $\{L_0, \dots, L_4\}$.
2. **Inter-Arrival Dynamics & Burst Jitter:** Log-transformed IAT sum, mean, std, relative variance, and temporal gradients.
3. **Opus 20ms NetEQ Harmonics:** Residual timing offsets relative to periodic speech framing ($\Delta t_i \pmod{0.020}$).
4. **Haar Wavelet Energy & Timing Ratios:** Multi-scale high-to-low frequency detail decomposition isolating transient keyframe bursts from smooth audio pacing.
5. **Markov State Transitions:** Transition probability matrix between small ($< 200\text{ B}$), voice ($100 - 350\text{ B}$), and video ($> 800\text{ B}$) packet states.
6. **Protocol Framing Discriminants:** Zoom SSCR 4-byte padding alignment (`pl_mod4_aligned`), WebRTC VP9 8-byte alignment (`pl_mod8_aligned`), and codec-specific envelope invariants (`pl_zoom_audio_envelope`, `pl_meet_audio_envelope`).

### View 2: Spatio-Temporal Sequence Embeddings
1. **4-Channel Tensor $(N, 4, 5)$:** Normalized Length, Normalized IAT, Packet Acceleration, and Cumulative Energy.
2. **2-Layer Bidirectional GRU:** Trained with AdamW to produce 64 non-linear temporal sequence embeddings.

---

## 3. Hierarchical Architecture (HMV-Stack)

1. **Stage 1 Modality Router:** Calibrated binary classifier separating Voice from Video with $> 96.8\%$ out-of-fold accuracy.
2. **Stage 2A & 2B Specialist Stacking Pools:** Two independent 4-model stacking ensembles (Scaled XGBoost, LightGBM, CatBoost, ExtraTrees) trained on Voice and Video subsets.
3. **Stage 3 Bayesian Soft-Gating:** Computes joint posterior probability $P(y=c) = P(\text{Modality}) \times P(c \mid \text{Modality})$.
4. **Stage 4 Focal Boundary Micro-Resolvers:** Dedicated high-resolution classifiers targeting ambiguous decision boundaries (`Zoom_video` vs `GoogleMeet_video` and `Zoom_voice` vs `Discord_voice`).
5. **Stage 5 Multi-Start Nelder-Mead Metric Calibration & 5-Pass TTA:** Multi-start direct optimization of Macro-F1 probability multipliers combined with 5-pass micro-jitter Test-Time Augmentation.

---

## 4. Empirical Evaluation Results ($N = 1,000,000$)

* **15-Fold Macro-F1 Score:** **`0.9259`**
* **Balanced Accuracy:** **`0.9258`**
* **Overall Accuracy:** **`92.58%`** ($925,800 / 1,000,000\text{ correct}$)

### Classification Report

| Target Class | Modality | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `WhatsApp_voice` | Voice | **0.9914** | **0.9889** | **0.9901** | 100,000 |
| `Messenger_voice` | Voice | **0.9900** | **0.9733** | **0.9816** | 100,000 |
| `Discord_voice` | Voice | **0.9696** | **0.9777** | **0.9736** | 100,000 |
| `Messenger_video` | Video | **0.9776** | **0.9704** | **0.9740** | 100,000 |
| `WhatsApp_video` | Video | **0.9537** | **0.9665** | **0.9601** | 100,000 |
| `Discord_video` | Video | **0.9593** | **0.9257** | **0.9422** | 100,000 |
| `Zoom_voice` | Voice | **0.8068** | **0.9299** | **0.8640** | 100,000 |
| `GoogleMeet_video` | Video | **0.8794** | **0.8597** | **0.8694** | 100,000 |
| `Zoom_video` | Video | **0.8333** | **0.8760** | **0.8541** | 100,000 |
| `GoogleMeet_voice` | Voice | **0.9199** | **0.7897** | **0.8498** | 100,000 |
| **Macro Average** | - | **0.9281** | **0.9258** | **0.9259** | **1,000,000** |

---

## 5. 6-Point Forensic Overfitting Verification

| Test Number | Audit Description | Threshold | Measured Result | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | Y-Randomization Null Test | F1 $\approx 0.1000$ | **`0.0998`** (Acc: 10.00%) | **PASSED** |
| **Test 2** | 15-Fold Stability Variance | Std Dev $\sigma < 0.0100$ | **Mean `0.9222`, $\sigma = 0.0070$** | **PASSED** |
| **Test 3** | Train-Test Data Contamination | 0 Overlapping Rows | **`0` Overlapping Rows** | **PASSED** |
| **Test 4** | Modality Router Stability | Binary Acc $> 95.0\%$ | **`96.84%`** | **PASSED** |
| **Test 5** | Test Shannon Entropy | Mean $< 1.0\text{ bit}$ | **Mean `0.2929 bits` ($88.1\%$ Confident)** | **PASSED** |
| **Test 6** | Train-Val Generalization Gap | Gap $\Delta < 0.0800$ | **Train `0.9849`, Val `0.9287`, Gap `0.0561`** | **PASSED** |

---

## 6. Artifact Index

* 📄 `results/submission.csv`: Final competition test set predictions ($N = 327$).
* 📄 `results/predictions.csv`: 10-class probability distribution matrix ($N = 327$).
* 📄 `results/confusion_matrix.png`: 10-class confusion matrix on 1,000,000 validation flows.
* 📄 `results/overfitting_audit_charts.png`: 6-point forensic audit visualization.
* 📄 `data/DATA_SOURCES.md`: Full citation and data extraction methodology from Inha University DNS Lab and MIRAGE-AppAct-2024.
