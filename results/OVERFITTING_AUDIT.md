# Forensic 6-Point Overfitting and Generalization Audit Report

This document reports the empirical verification and generalization audit conducted on the **Hierarchical Multi-View Stacking (HMV-Stack)** model trained on the expanded master dataset ($N = 10,500$ flows, 10 balanced classes).

---

## 1. Executive Summary: 6-Point Verification Matrix

| Audit Checkpoint | Metric Evaluated | Target Threshold | Measured Value | Verification Result |
| :--- | :--- | :---: | :---: | :---: |
| **1. Y-Randomization Null Test** | Permuted Label Macro-F1 | $\approx 0.1000$ (Random Chance) | **0.0993** (9.94% Acc) | PASSED (Zero Leakage) |
| **2. 15-Fold Stability Variance** | Fold Standard Deviation | $\sigma < 0.0200$ | **$\sigma = 0.0063$** (0.63%) | PASSED (Ultra-Stable) |
| **3. Test Set Isolation Audit** | Exact Row Duplicates | 0 Overlapping Rows | **0 Duplicates** | PASSED (Strict Isolation) |
| **4. Modality Router Generalization**| 15-Fold Binary CV Accuracy | $> 95.0\%$ | **96.84%** (10,168 / 10,500) | PASSED (Robust Routing) |
| **5. Test Prediction Entropy** | Mean Shannon Entropy | $< 1.50\text{ bits}$ | **0.2924 bits** (89.6% < 1.0b) | PASSED (High Confidence) |
| **6. Train-Val Generalization Gap** | Train F1 vs. Out-of-Fold Val F1 | $\Delta < 0.1000$ | **$\Delta = 0.0565$** (0.9842 vs 0.9277)| PASSED (Optimal Gap) |

---

## 2. Checkpoint Details & Verification Methodology

### Checkpoint 1: Y-Randomization Null Hypothesis Test
* **Methodology:** Targets $y$ are randomly permuted across the 10,500 flows, breaking real transport relationships while preserving marginal feature distributions.
* **Result:** Model Macro-F1 collapses to **0.0993** (9.94% accuracy, baseline random guess is 10.00%), confirming that the model learns real physical side-channel patterns rather than synthetic artifacts.

### Checkpoint 2: 15-Fold Cross-Validation Stability
* **Methodology:** 15 distinct stratified training/validation folds across 3 seeds (Seeds 42, 100, 2026).
* **Result:** Mean validation Macro-F1 is **0.9214** with a standard deviation of **$\sigma = 0.0063$** (0.63%), demonstrating that the stacking architecture generalizes uniformly across unseen data splits.

### Checkpoint 3: Test Set Contamination Audit
* **Methodology:** Pairwise Euclidean and exact coordinate hashing between Train ($N = 10,500$) and Test ($N = 327$).
* **Result:** Exactly 0 overlapping rows found between the training set and the test set.

### Checkpoint 4: Modality Router Generalization
* **Methodology:** Out-of-fold evaluation of Stage 1 binary routing (Voice vs. Video).
* **Result:** Achieved **96.84% accuracy** (10,168 / 10,500 correct classifications), ensuring that modal confusion does not propagate into downstream specialist pools.

### Checkpoint 5: Test Prediction Shannon Entropy
* **Methodology:** Shannon entropy computed on 10-class predicted probability distributions across the 327 unseen test samples:
  $$H(p) = -\sum_{i=1}^{10} p_i \log_2(p_i)$$
* **Result:** Mean test prediction entropy is **0.2924 bits** (maximum possible is 3.32 bits). **89.6% of test samples have entropy $< 1.0\text{ bit}$**, confirming sharp, high-confidence decision margins.

### Checkpoint 6: Train-Validation Generalization Gap
* **Methodology:** Gap $\Delta = \text{Train Macro-F1} - \text{Val Macro-F1}$.
* **Result:** Full training F1 is **0.9842**, out-of-fold validation F1 is **0.9277**, yielding a generalization gap of **$\Delta = 0.0565$** (well below the 0.1000 threshold).
