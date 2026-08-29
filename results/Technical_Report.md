# Comprehensive Technical Evaluation Report: HMV-Stack Pipeline

This technical report details the experimental methodology, model architecture, feature space, and 15-fold cross-validation performance of the **Hierarchical Multi-View Stacking (HMV-Stack)** model on the expanded **Master Dataset (N = 10,500 flows)**.

---

## 1. Master Leaderboard (15-Fold Cross-Validation, N = 10,500)

| Rank | Model Architecture | 15-Fold Macro-F1 | Balanced Accuracy | Overall Accuracy | Exact Flows Correct |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Hierarchical Multi-View Stacking (HMV-Stack, N=10.5k)** | **0.9374** | **0.9374** | **93.74%** | **9,843 / 10,500** |
| **2** | Hierarchical Multi-View Stacking (HMV-Stack, N=5.1k) | 0.9135 | 0.9138 | 91.40% | 4,730 / 5,175 |
| **3** | Feature-Augmented Stacking + Resolvers + TTA | 0.8867 | 0.8869 | 88.70% | 4,590 / 5,175 |
| **4** | Anti-Overfitting Multi-Seed Bagged Stacking | 0.8853 | 0.8864 | 88.66% | 4,588 / 5,175 |
| **5** | 3-Stage Selected Invariants Stacking | 0.8853 | 0.8858 | 88.60% | 4,585 / 5,175 |
| **6** | Siamese Residual SE-Encoder Metric Stacking | 0.8820 | 0.8828 | 88.29% | 4,569 / 5,175 |
| **7** | Standalone Tuned XGBoost (depth=7) | 0.8666 | 0.8671 | 86.74% | 4,489 / 5,175 |
| **8** | Standalone Tuned LightGBM (leaves=31) | 0.8647 | 0.8652 | 86.57% | 4,480 / 5,175 |

---

## 2. 10-Class Classification Report (HMV-Stack on N = 10,500)

| Class Name | Modality | Precision | Recall | F1-Score | Support | Correct Flows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `WhatsApp_voice` | Voice | 0.9943 | 0.9981 | 0.9962 | 1,050 | 1,048 |
| `Messenger_voice` | Voice | 0.9856 | 0.9752 | 0.9804 | 1,050 | 1,024 |
| `Messenger_video` | Video | 0.9761 | 0.9743 | 0.9752 | 1,050 | 1,023 |
| `WhatsApp_video` | Video | 0.9610 | 0.9867 | 0.9737 | 1,050 | 1,036 |
| `Discord_video` | Video | 0.9648 | 0.9648 | 0.9648 | 1,050 | 1,013 |
| `Discord_voice` | Voice | 0.9674 | 0.9610 | 0.9642 | 1,050 | 1,009 |
| `GoogleMeet_video` | Video | 0.9072 | 0.9029 | 0.9050 | 1,050 | 948 |
| `Zoom_video` | Video | 0.9116 | 0.8933 | 0.9024 | 1,050 | 938 |
| `GoogleMeet_voice` | Voice | 0.8621 | 0.8571 | 0.8596 | 1,050 | 900 |
| `Zoom_voice` | Voice | 0.8449 | 0.8610 | 0.8528 | 1,050 | 904 |
| **Exact Overall Average** | - | **0.9375** | **0.9374** | **0.9374** | **10,500** | **9,843** |

---

## 3. Key Pipeline Components

1. **Multi-View Representation (118 Features):**
   - **View 1 (54 Domain Invariants):** Haar wavelets, Opus 20ms NetEQ acoustic harmonics, first-order Markov transitions, dispersion moments, and byte rates.
   - **View 2 (Spatio-Temporal Tensor):** 4-channel tensor ($N, 4, 5$) processed by a 2-layer Bidirectional GRU yielding 64 temporal embeddings.
2. **Stage 1 Modality Router:** Calibrated binary GBDT router separating Voice vs. Video with **96.84% accuracy**.
3. **Stage 2 Modality Specialist Pools:** Two 4-model stacking pools (XGBoost, LightGBM, CatBoost, ExtraTrees) dedicated separately to Voice ($N = 5,250$) and Video ($N = 5,250$).
4. **Stage 3 Bayesian Soft-Gating:** Smooth probability fusion preventing hard threshold errors:
   $$P(y = c) = P(\text{Modality}) \times P(c \mid \text{Modality})$$
5. **Stage 4 Boundary Focal Micro-Resolvers:** Focused resolvers ($\gamma = 2.0$) targeting `Zoom_video` vs `GoogleMeet_video` and `Zoom_voice` vs `Discord_voice`.
6. **Stage 5 Nelder-Mead Calibration & 5-Pass TTA:** Metric multiplier optimization + micro-jitter test-time augmentation.
