# Technical Evaluation Report: Encrypted Network Traffic Classification

## 1. Overview and Methodology
This report documents the performance of the Stacking Model with Resolvers evaluated on the expanded master dataset (10 application/modality classes, N = 5,175 total flows). The architecture was evaluated under strict 15-fold cross-validation across three random seeds (42, 100, 2026).

## 2. Master Model Performance

| Metric | Score (15-Fold CV) | Exact Correct Flows | Support |
| :--- | :---: | :---: | :---: |
| **Macro-F1 Score** | **0.8867** | - | 5,175 |
| **Balanced Accuracy** | **0.8869** | - | 5,175 |
| **Overall Accuracy** | **88.70%** | **4,590 / 5,175** | 5,175 |

## 3. 10-Class Classification Report (N = 5,175 Flows)

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| WhatsApp_voice | 0.9904 | 0.9904 | 0.9904 | 520 |
| WhatsApp_video | 0.9465 | 0.9668 | 0.9565 | 512 |
| Messenger_voice | 0.9553 | 0.9572 | 0.9563 | 514 |
| Messenger_video | 0.9237 | 0.9237 | 0.9237 | 511 |
| GoogleMeet_voice | 0.9159 | 0.9212 | 0.9185 | 520 |
| Discord_video | 0.9250 | 0.8750 | 0.8993 | 536 |
| Discord_voice | 0.8462 | 0.9131 | 0.8784 | 518 |
| Zoom_voice | 0.8758 | 0.8135 | 0.8435 | 520 |
| GoogleMeet_video | 0.7875 | 0.7891 | 0.7883 | 512 |
| Zoom_video | 0.7063 | 0.7188 | 0.7125 | 512 |
| **Overall / Macro Average** | **0.8873** | **0.8869** | **0.8867** | **5,175** |

## 4. Key Architectural Mechanisms
1. **56 Feature Invariants**: 2-level Discrete Haar Wavelet (DWT) energy ratios, codec clump-to-silence ratios, log inter-arrival time moments, and payload variance ratios.
2. **Level-1 Base Estimators**: 15-fold out-of-fold probability vectors generated across LightGBM, XGBoost, and CatBoost.
3. **Level-2 Meta-Learner**: ElasticNet-regularized Logistic Regression (C=0.20, L1-ratio=0.20) operating on 86 meta-features (30 probabilities + 56 physical features).
4. **Pairwise Conflict Resolvers**: Dedicated micro-classifiers triggered on high-uncertainty samples (margin < 0.15) for Zoom vs Google Meet video and voice traffic.
5. **Test-Time Augmentation (TTA)**: 5-pass micro-jitter averaging at inference time to handle hardware timing variance.
