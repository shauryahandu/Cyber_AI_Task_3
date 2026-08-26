# Benchmark Report: Standalone Models Comparison

## 1. Experimental Protocol
Individual model families were trained and tested on the 56 engineered features across the full dataset (N = 3,325 flows) under identical 15-fold cross-validation splits.

## 2. Standalone Model Comparison

| Model Family | Algorithm Type | Macro-F1 | Overall Accuracy | Key Hyperparameters |
| :--- | :--- | :---: | :---: | :--- |
| LightGBM | Gradient Boosted Decision Trees (Leaf-wise) | 0.8508 | 85.38% | num_leaves=31, lr=0.08, colsample=0.80, subsample=0.90 |
| XGBoost | Gradient Boosted Decision Trees (Depth-wise) | 0.8396 | 84.18% | max_depth=5, lr=0.08, colsample=0.75, subsample=0.90 |
| TabPFN v3 | In-Context Attention Transformer | 0.8355 | 83.52% | temperature=0.80, n_estimators=12 |
| CatBoost | Symmetric / Oblivious Decision Trees | 0.8256 | 82.68% | depth=5, lr=0.08, iterations=350 |

## 3. Findings
1. LightGBM demonstrated the strongest standalone performance among individual models due to its leaf-wise tree splitting on non-linear inter-arrival timing boundaries.
2. TabPFN v3 provided high complementary diversity when integrated into the Level-2 stacking meta-learner, improving complex boundary separation for Zoom and Google Meet flows.
