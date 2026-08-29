# Encrypted Real-Time Communication Traffic Classification

Official implementation of the **Hierarchical Multi-View Stacking (HMV-Stack)** architecture for classifying encrypted WebRTC real-time communication (RTC) network flows using only **5 consecutive packets** across 10 balanced classes ($N = 10,500$).

Dataset collected at the **Distributed and Networked Systems Laboratory (DNS Lab), Inha University** for the **CyberAI Cup 2026 (Task 3: RTC)** held in conjunction with **ICONIP 2026** and **AICS 2026** ([https://csmining.org/cyberai2026/](https://csmining.org/cyberai2026/)).

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

| Class Name | Modality | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `WhatsApp_voice` | Voice | 0.9943 | 0.9981 | 0.9962 | 1,050 |
| `Messenger_voice` | Voice | 0.9856 | 0.9752 | 0.9804 | 1,050 |
| `Messenger_video` | Video | 0.9761 | 0.9743 | 0.9752 | 1,050 |
| `WhatsApp_video` | Video | 0.9610 | 0.9867 | 0.9737 | 1,050 |
| `Discord_video` | Video | 0.9648 | 0.9648 | 0.9648 | 1,050 |
| `Discord_voice` | Voice | 0.9674 | 0.9610 | 0.9642 | 1,050 |
| `GoogleMeet_video` | Video | 0.9072 | 0.9029 | 0.9050 | 1,050 |
| `Zoom_video` | Video | 0.9116 | 0.8933 | 0.9024 | 1,050 |
| `GoogleMeet_voice` | Voice | 0.8621 | 0.8571 | 0.8596 | 1,050 |
| `Zoom_voice` | Voice | 0.8449 | 0.8610 | 0.8528 | 1,050 |
| **Exact Overall Average** | - | **0.9375** | **0.9374** | **0.9374** | **10,500** |

---

## 3. Architecture Specification & Flow

Detailed breakdown available in [`architecture/ARCHITECTURE_FLOW.md`](architecture/ARCHITECTURE_FLOW.md).

![Architecture Flow Diagram](architecture/architecture_flow_diagram.png)

1. **Multi-View Feature Fusion (118 Features):**
   - **View 1 (54 Handcrafted Invariants):** Haar wavelet energy ratios, Opus 20ms NetEQ acoustic harmonics, first-order Markov transitions.
   - **View 2 (Spatio-Temporal Sequence):** 2-layer Bidirectional GRU trained on a 4-channel tensor (Length, IAT, Acceleration, Cumulative Energy).
2. **Stage 1 Modality Router:** Calibrated binary GBDT classifier separating Voice vs Video with **96.84% accuracy**.
3. **Stage 2A & 2B Modality Specialist Pools:** Two 4-model stacking ensembles (XGBoost, LightGBM, CatBoost, ExtraTrees) dedicated separately to Voice ($N=5,250$) and Video ($N=5,250$).
4. **Stage 3 Bayesian Soft-Gating:** Smooth joint posterior probability calculation: $P(y = c) = P(\text{Modality}) \times P(c \mid \text{Modality})$.
5. **Stage 4 Boundary Focal Resolvers:** Resolvers ($\gamma = 2.0$) targeting `Zoom_video` vs `GoogleMeet_video` and `Zoom_voice` vs `Discord_voice`.
6. **Stage 5 Calibration & 5-Pass TTA:** Nelder-Mead Macro-F1 multiplier optimization + micro-jitter test-time augmentation.

---

## 4. Overfitting & Generalization Audit

Full audit report available in [`results/OVERFITTING_AUDIT.md`](results/OVERFITTING_AUDIT.md).

| Audit Checkpoint | Metric Measured | Result | Status |
| :--- | :--- | :---: | :---: |
| **Y-Randomization Null Test** | Permuted Label Macro-F1 | 0.0993 (9.94% Acc) | PASSED (Zero Leakage) |
| **15-Fold Stability Variance** | Fold Standard Deviation | Sigma = 0.0063 (0.63%) | PASSED (Ultra-Stable) |
| **Test Contamination Audit** | Exact Row Overlap | 0 Duplicate Rows | PASSED (Strict Isolation) |
| **Modality Router Generalization**| 15-Fold Binary CV Accuracy | 96.84% (10,168 / 10,500) | PASSED (Robust Gating) |
| **Test Prediction Entropy** | Mean Shannon Entropy | 0.2924 bits (89.6% < 1.0b) | PASSED (High Confidence) |
| **Train-Val Generalization Gap** | Train F1 vs. Out-of-Fold Val F1 | Delta = 0.0565 (0.9842 vs 0.9277)| PASSED (< 0.1000) |

---

## 5. Literature References & Textbook

### Reference Textbook
* **Title:** *Machine Learning and Security in Encrypted Network Traffic Analysis*
* **Publisher:** Springer Nature Singapore
* **ISBN-13:** 978-981-97-7737-2 | **ISBN-10:** 981-97-7737-2
* **DOI / Link:** [https://doi.org/10.1007/978-981-97-7737-2](https://doi.org/10.1007/978-981-97-7737-2)

### Academic Research Papers
* **ECHO Framework:** *Early Encrypted Traffic Classification using Few-Packet Dynamics* (IEEE/ACM Transactions on Networking, [arXiv:2403.18121](https://arxiv.org/abs/2403.18121)).
* **Caps-LSTM:** *Hierarchical Spatial-Temporal Modeling for Encrypted Network Traffic* ([MDPI Electronics](https://doi.org/10.3390/electronics12102205)).
* **FlowPrint:** *Semi-Supervised Mobile App Fingerprinting on Encrypted Traffic* ([NDSS Symposium](https://www.ndss-symposium.org/)).
* **Deep Learning for Encrypted Traffic Survey:** *Deep Learning Methods for Encrypted Traffic Classification: A Comprehensive Survey* ([IEEE Communications Surveys & Tutorials](https://doi.org/10.1109/COMST.2023.3275464)).
* **WebRTC Traffic Characterization:** *Real-Time Communications (RTC) Encrypted Traffic Classification under WebRTC Protocols* (ACM CoNEXT / IEEE INFOCOM).
* **TabArena Benchmark:** *A Benchmark for Tabular Machine Learning* ([Hugging Face / NeurIPS 2024](https://huggingface.co/spaces/TabArena/leaderboard)).
* **AppClassNet & CipherSpectrum:** *Modern Benchmark Datasets for TLS 1.3 & QUIC Encrypted Traffic Classification* ([SOTA2](https://sota2.com/dataset/appclassnet)).

---

## 6. Repository Structure

```
├── architecture/
│   ├── ARCHITECTURE_FLOW.md       # Technical pipeline specification & research citations
│   └── architecture_flow_diagram.png # 5-stage architecture flow diagram
├── data/
│   ├── DATA_SOURCES.md            # Inha University lab provenance, mining procedure, links, audits
│   ├── Training_set.csv           # Original baseline training set (N = 1,285)
│   ├── Testing_set.csv            # Original competition test set (N = 327)
│   └── Augmented_Training_set.csv  # Verified 10-class master dataset (N = 10,500)
├── feature_engineering/
│   ├── FEATURE_DICTIONARY.md      # Full feature definitions and mathematical formulas
│   └── plots/                     # High-resolution feature distribution plots
├── src/
│   ├── audio_to_csv.py            # Audio WAV framing & flow conversion utility
│   ├── data_mining.py             # PCAP parsing & 5-packet window extraction routine
│   ├── features.py                # Transport invariant extraction & sequence tensor builder
│   ├── model.py                   # Bi-GRU sequence embedding neural backbone
│   ├── train_and_evaluate.py      # 15-fold cross-validation training and evaluation
│   ├── verify_overfitting.py      # 6-point forensic overfitting & generalization audit
│   └── predict.py                 # Standalone inference generator
├── results/
│   ├── submission.csv             # Standard competition submission file (N = 327)
│   ├── predictions.csv            # Predicted labels and probability distributions (N = 327)
│   ├── Testing_set_with_predictions.csv # Test features merged with predicted labels (N = 327)
│   ├── confusion_matrix.png       # 10-class confusion matrix heatmap
│   ├── overfitting_audit_charts.png # 15-fold stability and test entropy distributions
│   ├── OVERFITTING_AUDIT.md       # 6-point forensic audit report
│   └── Technical_Report.md        # Comprehensive technical report
├── requirements.txt                # Pinned dependencies
└── README.md                       # Documentation, research citations, & data source links
```

---

## 7. Usage & Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run PCAP data mining and flow extraction
python3 src/data_mining.py data/raw_captures data/Augmented_Training_set.csv

# Run audio WAV to packet flow conversion
python3 src/audio_to_csv.py data/audio_raw data/audio_extracted_flows.csv

# Run 15-fold cross-validation & generate predictions
python3 src/train_and_evaluate.py

# Run the 6-point overfitting & generalization audit
python3 src/verify_overfitting.py

# Run standalone inference on test traffic
python3 src/predict.py data/Testing_set.csv results/predictions.csv
```
