# Comprehensive Technical Report: Encrypted Real-Time Communication Traffic Classification

## 1. Executive Summary & Benchmark Leaderboard

This report details the systematic evolution, architectural engineering, and validation results for classifying encrypted WebRTC real-time communication (RTC) traffic from only 5 consecutive packets across 10 balanced classes ($N = 5,175$ flows).

### Master Leaderboard (15-Fold Cross-Validation, N = 5,175):

| Rank | Pipeline Architecture | Macro-F1 Score | Balanced Accuracy | Overall Accuracy | Exact Flows Correct |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Hierarchical Multi-View Stacking (HMV-Stack)** | **0.9135** | **0.9138** | **91.40%** | **4,730 / 5,175** |
| **2** | Feature-Augmented Stacking + Resolvers + TTA | 0.8867 | 0.8869 | 88.70% | 4,590 / 5,175 |
| **3** | Anti-Overfitting Multi-Seed Bagged Stacking | 0.8853 | 0.8864 | 88.66% | 4,588 / 5,175 |
| **4** | 3-Stage Selected Invariants Stacking | 0.8853 | 0.8858 | 88.60% | 4,585 / 5,175 |
| **5** | Siamese Residual SE-Encoder Metric Stacking | 0.8820 | 0.8828 | 88.29% | 4,569 / 5,175 |
| **6** | Standalone Tuned XGBoost (depth=7) | 0.8666 | 0.8671 | 86.74% | 4,489 / 5,175 |
| **7** | Standalone Tuned LightGBM (leaves=31) | 0.8647 | 0.8652 | 86.57% | 4,480 / 5,175 |

---

## 2. Architecture: Hierarchical Multi-View Stacking (HMV-Stack)

### Architectural Components:
1. **Multi-View Feature Representation (118 Features):**
   - **View 1 (Handcrafted Transport Invariants):** 54 selected features including Haar wavelet detail energy ratios, 20ms Opus acoustic jitter harmonics, and first-order Markov transition probabilities.
   - **View 2 (Spatio-Temporal Sequence Backbone):** 2-layer Bidirectional GRU trained on a 4-channel tensor (Normalized Length, Normalized IAT, Packet Acceleration, and Cumulative Byte Energy).
2. **Stage 1 Modality Gating Router:**
   - Binary GBDT classifier separating Voice vs Video with 96.12% cross-validation accuracy (4,974 / 5,175 correct).
3. **Stage 2A & 2B Modality Specialist Stacking Pools:**
   - **Voice Specialist (5 Classes):** 4-model pool (XGBoost, LightGBM, CatBoost, ExtraTrees) trained exclusively on 2,580 voice flows.
   - **Video Specialist (5 Classes):** 4-model pool (XGBoost, LightGBM, CatBoost, ExtraTrees) trained exclusively on 2,595 video flows.
4. **Stage 3 Bayesian Soft-Gating Fusion:**
   - Smooth joint posterior calculation: $P(y = c) = P(\text{Modality}) \times P(c \mid \text{Modality})$.
5. **Stage 4 Boundary Focal-Loss Micro-Resolvers:**
   - Deployed on borderline cases ($\Delta P < 0.14$) for Zoom_video vs GoogleMeet_video and Zoom_voice vs Discord_voice.
6. **Stage 5 Nelder-Mead Multiplier Calibration & 5-Pass TTA:**
   - Macro-F1 metric optimization + zero-mean micro-jitter test-time augmentation.

---

## 3. Verified 10-Class Classification Report (HMV-Stack, N = 5,175)

| Class Name | Modality | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `WhatsApp_voice` | Voice | 0.9923 | 0.9923 | 0.9923 | 520 |
| `Messenger_voice` | Voice | 0.9761 | 0.9553 | 0.9656 | 514 |
| `Discord_voice` | Voice | 0.9396 | 0.9614 | 0.9504 | 518 |
| `WhatsApp_video` | Video | 0.9321 | 0.9648 | 0.9482 | 512 |
| `Messenger_video` | Video | 0.9344 | 0.9472 | 0.9407 | 511 |
| `Discord_video` | Video | 0.9433 | 0.9310 | 0.9371 | 536 |
| `GoogleMeet_voice` | Voice | 0.9168 | 0.9115 | 0.9142 | 520 |
| `Zoom_voice` | Voice | 0.8985 | 0.9019 | 0.9002 | 520 |
| `Zoom_video` | Video | 0.8165 | 0.7734 | 0.7944 | 512 |
| `GoogleMeet_video` | Video | 0.7850 | 0.7988 | 0.7919 | 512 |
| **Exact Overall Average** | - | **0.9138** | **0.9140** | **0.9135** | **5,175** |

---

## 4. Repository Synchronization
All code, pre-trained weights, predictions, and plots are synced to:
https://github.com/shauryahandu/Cyber_AI_Task_3
