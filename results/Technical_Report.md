# Technical Evaluation Report: Encrypted Network Traffic Classification

## 1. Overview and Methodology
This report documents the performance of machine learning architectures evaluated on the encrypted UDP traffic classification benchmark (10 application/modality classes, N = 3,325 total flows). All models were evaluated under strict 15-fold cross-validation across three random seeds (42, 100, 2026).

## 2. Model Performance Leaderboard

| Architecture | Macro-F1 | Balanced Accuracy | Overall Accuracy | Support |
| :--- | :---: | :---: | :---: | :---: |
| Feature-Augmented Stacking with Resolvers & TTA | 0.8644 | 0.8641 | 86.53% | 3,325 |
| Flat Feature-Augmented Stacking (ElasticNet) | 0.8618 | 0.8619 | 86.35% | 3,325 |
| Hierarchical Cascaded Stacking | 0.8609 | 0.8610 | 86.26% | 3,325 |
| Cascaded 3-Model Pipeline | 0.8595 | 0.8593 | 86.14% | 3,325 |
| Two-Stage Modality Routing Ensemble | 0.8572 | 0.8570 | 85.92% | 3,325 |
| Calibrated Blend | 0.8530 | 0.8538 | 85.59% | 3,325 |
| Standalone Tuned LightGBM | 0.8508 | 0.8513 | 85.38% | 3,325 |
| Standalone Tuned XGBoost | 0.8396 | 0.8401 | 84.18% | 3,325 |
| Standalone Tuned TabPFN v3 | 0.8355 | 0.8359 | 83.52% | 3,325 |
| Pre-Augmentation Baseline | 0.8147 | 0.8403 | 81.71% | 1,285 |

## 3. 10-Class Classification Report (Best Model: Stacking with Resolvers)

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| WhatsApp_voice | 0.9833 | 0.9833 | 0.9833 | 300 |
| WhatsApp_video | 0.9387 | 0.9636 | 0.9510 | 302 |
| GoogleMeet_voice | 0.9010 | 0.9100 | 0.9055 | 300 |
| Discord_video | 0.9043 | 0.8706 | 0.8872 | 456 |
| Messenger_video | 0.8800 | 0.8640 | 0.8720 | 331 |
| Messenger_voice | 0.8754 | 0.8605 | 0.8679 | 294 |
| Discord_voice | 0.8344 | 0.8858 | 0.8594 | 438 |
| Zoom_voice | 0.8153 | 0.7800 | 0.7973 | 300 |
| GoogleMeet_video | 0.8191 | 0.7947 | 0.8067 | 302 |
| Zoom_video | 0.7006 | 0.7285 | 0.7143 | 302 |
| **Overall / Macro Average** | **0.8652** | **0.8641** | **0.8644** | **3,325** |

## 4. Key Architectural Mechanisms
1. **Feature Engineering**: 56 domain invariants including 2-level Discrete Haar Wavelet (DWT) energy ratios, codec clump-to-silence ratios, log inter-arrival time moments, and payload variance ratios.
2. **Level-1 Base Estimators**: 15-fold out-of-fold probability vectors generated across LightGBM, XGBoost, CatBoost, and TabPFN v3.
3. **Level-2 Meta-Learner**: ElasticNet-regularized Logistic Regression (C=0.20, L1-ratio=0.20) operating on 96 meta-features (40 probabilities + 56 physical features).
4. **Pairwise Conflict Resolvers**: Dedicated micro-classifiers triggered on high-uncertainty samples (margin < 0.15) for Zoom vs Google Meet video and voice traffic.
5. **Test-Time Augmentation (TTA)**: 5-pass micro-jitter averaging at inference time to handle hardware timing variance.
