# Encrypted Real-Time Communication Traffic Classification

This repository contains the official implementation of the **Hierarchical Multi-View Stacking (HMV-Stack)** architecture for classifying encrypted WebRTC real-time communication (RTC) network flows using only **5 consecutive packets** per flow across 10 balanced classes ($N = 5,175$).

---

## 1. Benchmark Leaderboard

| Model Architecture | 15-Fold Macro-F1 | Balanced Accuracy | Overall Accuracy | Exact Flows Correct |
| :--- | :---: | :---: | :---: | :---: |
| **Hierarchical Multi-View Stacking (HMV-Stack)** | **0.9135** | **0.9138** | **91.40%** | **4,730 / 5,175** |
| Feature-Augmented Stacking + Resolvers + TTA | 0.8867 | 0.8869 | 88.70% | 4,590 / 5,175 |
| Anti-Overfitting Multi-Seed Bagged Stacking | 0.8853 | 0.8864 | 88.66% | 4,588 / 5,175 |
| 3-Stage Selected Invariants Stacking | 0.8853 | 0.8858 | 88.60% | 4,585 / 5,175 |
| Siamese Residual SE-Encoder Metric Stacking | 0.8820 | 0.8828 | 88.29% | 4,569 / 5,175 |
| Standalone Tuned XGBoost (depth=7) | 0.8666 | 0.8671 | 86.74% | 4,489 / 5,175 |
| Standalone Tuned LightGBM (leaves=31) | 0.8647 | 0.8652 | 86.57% | 4,480 / 5,175 |

---

## 2. 10-Class Classification Report (HMV-Stack)

| Class Name | Modality | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| WhatsApp_voice | Voice | 0.9923 | 0.9923 | 0.9923 | 520 |
| Messenger_voice | Voice | 0.9761 | 0.9553 | 0.9656 | 514 |
| Discord_voice | Voice | 0.9396 | 0.9614 | 0.9504 | 518 |
| WhatsApp_video | Video | 0.9321 | 0.9648 | 0.9482 | 512 |
| Messenger_video | Video | 0.9344 | 0.9472 | 0.9407 | 511 |
| Discord_video | Video | 0.9433 | 0.9310 | 0.9371 | 536 |
| GoogleMeet_voice | Voice | 0.9168 | 0.9115 | 0.9142 | 520 |
| Zoom_voice | Voice | 0.8985 | 0.9019 | 0.9002 | 520 |
| Zoom_video | Video | 0.8165 | 0.7734 | 0.7944 | 512 |
| GoogleMeet_video | Video | 0.7850 | 0.7988 | 0.7919 | 512 |
| **Exact Overall Average** | - | **0.9138** | **0.9140** | **0.9135** | **5,175** |

---

## 3. Architecture Design

1. **Multi-View Feature Representation (118 Features):**
   - **View 1 (Handcrafted Transport Invariants):** 54 selected non-redundant features including Haar wavelet detail energy ratios, 20ms Opus acoustic jitter harmonics, and first-order Markov transition probabilities.
   - **View 2 (Spatio-Temporal Sequence Backbone):** 2-layer Bidirectional GRU trained on a 4-channel tensor (Normalized Length, Normalized IAT, Packet Acceleration, and Cumulative Energy).
2. **Stage 1 Modality Gating Router:**
   - Binary GBDT classifier separating Voice vs Video with 96.12% cross-validation accuracy.
3. **Stage 2A & 2B Modality Specialist Stacking Pools:**
   - **Voice Specialist (5 Classes):** 4-model pool (XGBoost, LightGBM, CatBoost, ExtraTrees) trained on voice flows.
   - **Video Specialist (5 Classes):** 4-model pool (XGBoost, LightGBM, CatBoost, ExtraTrees) trained on video flows.
4. **Stage 3 Bayesian Soft-Gating Fusion:**
   - Smooth joint posterior calculation: $P(y = c) = P(\text{Modality}) \times P(c \mid \text{Modality})$.
5. **Stage 4 Boundary Focal-Loss Micro-Resolvers:**
   - Resolvers for Zoom_video vs GoogleMeet_video and Zoom_voice vs Discord_voice.
6. **Stage 5 Nelder-Mead Multiplier Calibration & 5-Pass TTA:**
   - Metric optimization + zero-mean micro-jitter test-time augmentation.

---

## 4. Repository Structure

```
├── data/
│   ├── Training_set.csv           # Original training set
│   ├── Testing_set.csv            # Original testing set (N = 327)
│   └── Augmented_Training_set.csv  # Verified master dataset (N = 5,175)
├── src/
│   ├── features.py                # Transport invariant extraction & sequence tensor builder
│   ├── model.py                   # Bi-GRU sequence embedding backbone
│   ├── train_and_evaluate.py      # 15-fold cross-validation training and evaluation
│   └── predict.py                 # Test inference script
├── results/
│   ├── predictions.csv            # Predictions and probability distributions for Testing_set.csv
│   ├── confusion_matrix.png       # 10-class confusion matrix heatmap
│   ├── overfitting_audit_charts.png # 15-fold stability and test entropy distributions
│   └── Technical_Report.md        # Comprehensive technical report
└── README.md
```

---

## 5. Usage & Reproducibility

### Installation
```bash
pip install -r requirements.txt
# Requires: torch, lightgbm, xgboost, catboost, scikit-learn, scipy, pandas, numpy, seaborn, matplotlib
```

### Run Cross-Validation & Generate Deliverables
```bash
python3 src/train_and_evaluate.py
```

### Run Inference on Unseen Traffic
```bash
python3 src/predict.py data/Testing_set.csv results/predictions.csv
```
