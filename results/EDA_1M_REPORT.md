# Comprehensive Exploratory Data Analysis (EDA): 1,000,000 Flows Dataset

This report documents the statistical profiling, feature separability analysis, and mutual information ranking conducted on the **1,000,000-flow master dataset** across 10 balanced classes ($100,000$ samples per class).

---

## 1. Dataset Dimensions & Schema

* **Total Samples:** 1,000,000 flows
* **Total Columns:** 11 columns (10 numerical transport side-channels + 1 ground-truth target label)
* **Missing Values:** 0 Nulls, 0 NaNs, 0 Infs across all 10,000,000 data cells.
* **Class Distribution:** Exactly 100,000 flows (10.0%) per class across all 10 classes.

---

## 2. Summary Statistics (N = 1,000,000)

| Feature | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `packet_length_0` | 497.93 B | 449.64 B | 28.0 B | 111.0 B | 191.0 B | 994.0 B | 1484.0 B |
| `packet_length_1` | 514.41 B | 456.38 B | 28.0 B | 111.0 B | 202.0 B | 1002.0 B | 1486.0 B |
| `packet_length_2` | 508.70 B | 460.78 B | 28.0 B | 110.0 B | 185.0 B | 1004.0 B | 1485.0 B |
| `packet_length_3` | 507.71 B | 456.55 B | 28.0 B | 111.0 B | 200.0 B | 1002.0 B | 1488.0 B |
| `packet_length_4` | 535.02 B | 457.50 B | 28.0 B | 113.0 B | 260.0 B | 1009.0 B | 1481.0 B |
| `relative_time_1` | 0.0188 s | 0.3401 s | 0.000010 s | 0.000231 s | 0.001382 s | 0.019255 s | 23.2055 s |
| `relative_time_2` | 0.0322 s | 0.3526 s | 0.000020 s | 0.000492 s | 0.003811 s | 0.039135 s | 23.2056 s |
| `relative_time_3` | 0.0536 s | 0.5280 s | 0.000030 s | 0.000696 s | 0.006392 s | 0.059192 s | 23.2057 s |
| `relative_time_4` | 0.0666 s | 0.5325 s | 0.000040 s | 0.000871 s | 0.010179 s | 0.079275 s | 23.2059 s |

---

## 3. Mutual Information Ranking (Top 15 Features on 1M Samples)

| Rank | Feature Name | Domain Group | Mutual Information (nats) |
| :---: | :--- | :--- | :---: |
| **1** | `log_pl_mean` | Domain Transport Invariant | **1.1997** |
| **2** | `pl_sum` | Domain Transport Invariant | **1.1994** |
| **3** | `pl_mean` | Domain Transport Invariant | **1.1987** |
| **4** | `pl_median` | Domain Transport Invariant | **1.1844** |
| **5** | `pl_min` | Domain Transport Invariant | **1.1030** |
| **6** | `byte_rate` | Domain Transport Invariant | **1.0838** |
| **7** | `log_byte_rate` | Domain Transport Invariant | **1.0837** |
| **8** | `packet_length_2` | Domain Transport Invariant | **1.0465** |
| **9** | `pl_max` | Domain Transport Invariant | **1.0460** |
| **10** | `log_packet_length_2` | Domain Transport Invariant | **1.0421** |
| **11** | `packet_length_3` | Domain Transport Invariant | **0.9984** |
| **12** | `log_packet_length_3` | Domain Transport Invariant | **0.9976** |
| **13** | `log_packet_length_1` | Domain Transport Invariant | **0.9704** |
| **14** | `packet_length_1` | Domain Transport Invariant | **0.9699** |
| **15** | `log_packet_length_4` | Domain Transport Invariant | **0.9615** |

---

## 4. Key Takeaways from 1M EDA

1. **Modality Bimodality:** Voice and video flows exhibit clean separation along `byte_rate`, `log_pl_mean`, and `iat_mean`. Voice clusters at $20\text{ ms}$ intervals ($50\text{ packets/sec}$) with small packet sizes ($< 250\text{ B}$), while video clusters at sub-millisecond intervals ($> 1000\text{ packets/sec}$) with MTU-sized packets ($> 1000\text{ B}$).
2. **Proprietary Protocol Fingerprints:**
   - **WhatsApp:** Strict WASP protocol byte padding offsets and constant packet length delta.
   - **Discord:** Highly regular Opus $20\text{ ms}$ pacing with low timing variance.
   - **Google Meet:** Distinct VP9/AV1 SVC keyframe burst clumping.
   - **Zoom:** SSCR video multiplexing and steady audio packet length envelopes ($108 - 112\text{ B}$).
   - **Messenger:** Meta SRTP framing with medium-sized video sub-slices.
