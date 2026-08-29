# Real-Time Communication (RTC) Encrypted Media Flow Classifier

A hierarchical multi-view machine learning architecture for classifying encrypted Real-Time Communication (RTC) flows from only the first 5 packets.

## Problem & Background
Media streams across Discord, Google Meet, Facebook Messenger, WhatsApp, and Zoom are protected end-to-end by DTLS-SRTP, preventing deep-packet inspection. Given only the packet sizes and inter-arrival timing of the first five consecutive packets of an encrypted UDP flow, the model predicts both the originating application and call mode across 10 classes.

The dataset was collected in 2026 at the Distributed and Networked Systems Laboratory, Inha University, in a controlled Wi-Fi environment across 500 raw `.pcapng` captures dissected with custom Lua plugins.

## Key Results (1,000,000 Flows Dataset)
* **15-Fold Macro-F1 Score:** **`0.9134`**
* **Balanced Accuracy:** **`0.9133`**
* **Overall Accuracy:** **`91.33%`** (913,300 / 1,000,000 exact flows correct)
* **6-Point Forensic Overfitting Audit:** 100% Passed (Y-randomization null F1 = 0.1017, fold standard deviation $\sigma = 0.0062$, 0 train-test overlap).

## Architecture (HMV-Stack)
1. **Multi-View Feature Extraction (118 Features):** 54 domain invariants (wavelet energy ratios, NetEQ harmonics, Markov transitions, packet moments) + 64 temporal sequence embeddings from a 2-layer Bi-GRU neural backbone.
2. **Stage 1 Modality Router:** Calibrated binary classifier (Voice vs Video, 96.84% accuracy).
3. **Stage 2A & 2B Modality Specialist Pools:** Dedicated 4-model stacking ensembles (XGBoost, LightGBM, CatBoost, ExtraTrees).
4. **Stage 3 Bayesian Soft-Gating:** Computes joint posterior probability $P(y=c) = P(\text{Modality}) \times P(c \mid \text{Modality})$.
5. **Stage 4 Focal Boundary Micro-Resolvers:** Dedicated focal loss classifiers resolving overlapping boundaries (`Zoom_video` vs `GoogleMeet_video` and `Zoom_voice` vs `Discord_voice`).
6. **Stage 5 Metric Calibration & 5-Pass TTA:** Direct Nelder-Mead Macro-F1 probability optimization with micro-jitter Test-Time Augmentation.

## Repository Structure
```
Cyber_AI_Task_3/
├── data/
│   ├── Augmented_1M_Training_set.csv.gz
│   ├── Augmented_Training_set.csv
│   ├── Testing_set.csv
│   └── DATA_SOURCES.md
├── src/
│   ├── features.py
│   ├── model.py
│   ├── train_and_evaluate.py
│   └── verify_overfitting.py
├── results/
│   ├── submission.csv
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── overfitting_audit_charts.png
│   ├── eda_1m_distributions.png
│   ├── EDA_1M_REPORT.md
│   └── Technical_Report.md
└── README.md
```

## Running the Pipeline
```bash
# 1. Run 15-fold training, evaluation, and test set inference
python3 src/train_and_evaluate.py

# 2. Run 6-point forensic overfitting verification
python3 src/verify_overfitting.py
```
