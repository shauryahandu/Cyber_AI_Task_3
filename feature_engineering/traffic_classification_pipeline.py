import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
sns.set_style("whitegrid")
INPUT_DIR = "/kaggle/input/your-dataset-name"
TRAIN_PATH = os.path.join(INPUT_DIR, "Augmented_Training_set.xlsx")
TEST_PATH = os.path.join(INPUT_DIR, "Testing_set.xlsx")
OUTPUT_DIR = "/kaggle/working"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "Augmented_Training_set.xlsx"
    TEST_PATH = "Testing_set.xlsx"
    OUTPUT_DIR = "."
train = pd.read_excel(TRAIN_PATH)
test = pd.read_excel(TEST_PATH)
fig, ax = plt.subplots(figsize=(10, 5))
train["label"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Class distribution (train)")
ax.set_ylabel("count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_class_balance.png", dpi=120)
plt.close()
pl_cols = [f"packet_length_{i}" for i in range(5)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
train.assign(mode=train["label"].str.split("_").str[1]).boxplot(
    column="packet_length_0", by="mode", ax=axes[0]
)
axes[0].set_title("packet_length_0 by call type")
train.boxplot(column="packet_length_0", by="label", ax=axes[1], rot=90)
axes[1].set_title("packet_length_0 by class")
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_packet_length_by_class.png", dpi=120)
plt.close()
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pl = [f"packet_length_{i}" for i in range(5)]
    for i in range(1, 5):
        df[f"iat_{i}"] = df[f"relative_time_{i}"] - df[f"relative_time_{i-1}"]
    for i in range(1, 5):
        df[f"pl_delta_{i}"] = df[f"packet_length_{i}"] - df[f"packet_length_{i-1}"]
    df["pl_mean"] = df[pl].mean(axis=1)
    df["pl_std"] = df[pl].std(axis=1)
    df["pl_min"] = df[pl].min(axis=1)
    df["pl_max"] = df[pl].max(axis=1)
    df["pl_range"] = df["pl_max"] - df["pl_min"]
    df["pl_median"] = df[pl].median(axis=1)
    df["pl_sum"] = df[pl].sum(axis=1)
    df["pl_skew"] = df[pl].skew(axis=1)
    df["pl_cv"] = df["pl_std"] / (df["pl_mean"] + 1e-9)
    iat_cols = [f"iat_{i}" for i in range(1, 5)]
    df["iat_mean"] = df[iat_cols].mean(axis=1)
    df["iat_std"] = df[iat_cols].std(axis=1)
    df["iat_max"] = df[iat_cols].max(axis=1)
    df["iat_min"] = df[iat_cols].min(axis=1)
    df["iat_sum"] = df["relative_time_4"]
    for c in iat_cols + ["iat_sum"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    for c in pl:
        df[f"log_{c}"] = np.log1p(df[c])
    df["n_small_pkts"] = (df[pl] < 100).sum(axis=1)
    df["n_large_pkts"] = (df[pl] > 1200).sum(axis=1)
    rest_cols = [f"packet_length_{i}" for i in range(1, 5)]
    df["pl0_to_mean_rest"] = df["packet_length_0"] / (df[rest_cols].mean(axis=1) + 1e-9)
    df["n_fast_iat"] = (df[iat_cols] < 0.001).sum(axis=1)
    df["byte_rate"] = df["pl_sum"] / (df["relative_time_4"] + 1e-6)
    return df
train_fe = engineer_features(train)
test_fe = engineer_features(test)
assert train_fe.drop(columns=["label"]).isnull().sum().sum() == 0
assert test_fe.isnull().sum().sum() == 0
le = LabelEncoder()
y = le.fit_transform(train_fe["label"])
X = train_fe.drop(columns=["label"])
X_test = test_fe[X.columns]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
probe = LGBMClassifier(
    n_estimators=300, learning_rate=0.08, random_state=RANDOM_STATE,
    verbosity=-1, n_jobs=-1,
)
probe.fit(X, y)
importances = pd.Series(probe.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
importances.head(20).sort_values().plot(kind="barh", ax=ax, color="teal")
ax.set_title("Top 20 feature importances (LightGBM)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=120)
plt.close()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
models = {
    "LogReg": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
    "KNN": Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "SVM_RBF": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale"))]),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08, random_state=RANDOM_STATE,
        eval_metric="mlogloss", n_jobs=-1, verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.9,
        colsample_bytree=0.8, min_child_samples=10, random_state=RANDOM_STATE,
        verbosity=-1, n_jobs=-1,
    ),
}
comparison_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=["f1_macro", "f1_weighted", "accuracy"], n_jobs=1)
    comparison_rows.append({
        "model": name,
        "f1_macro": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std(),
        "f1_weighted": scores["test_f1_weighted"].mean(),
        "accuracy": scores["test_accuracy"].mean(),
    })
          f"(+-{comparison_rows[-1]['f1_macro_std']:.4f})  "
          f"F1-weighted={comparison_rows[-1]['f1_weighted']:.4f}  "
          f"Acc={comparison_rows[-1]['accuracy']:.4f}")
comparison_df = pd.DataFrame(comparison_rows).sort_values("f1_macro", ascending=False)
comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 5))
comparison_df.set_index("model")["f1_macro"].plot(kind="barh", ax=ax, color="darkorange")
ax.set_xlabel("F1-macro (5-fold CV)")
ax.set_title("Model comparison")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=120)
plt.close()
best_model_name = comparison_df.iloc[0]["model"]
final_params = dict(
    n_estimators=300,
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=RANDOM_STATE,
    verbosity=-1,
    n_jobs=-1,
)
final_model = LGBMClassifier(**final_params)
oof_preds = cross_val_predict(final_model, X, y, cv=cv, n_jobs=1)
oof_f1_macro = f1_score(y, oof_preds, average="macro")
oof_f1_weighted = f1_score(y, oof_preds, average="weighted")
cm = confusion_matrix(y, oof_preds)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
)
ax.set_title(f"Confusion matrix (OOF) — LightGBM, F1-macro={oof_f1_macro:.3f}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()
final_model.fit(X, y)
test_preds_encoded = final_model.predict(X_test)
test_preds = le.inverse_transform(test_preds_encoded)
test_proba = final_model.predict_proba(X_test)
submission = pd.DataFrame({
    "id": np.arange(len(test)),
    "label": test_preds,
})
submission.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
proba_df = pd.DataFrame(test_proba, columns=le.classes_)
proba_df.insert(0, "id", np.arange(len(test)))
proba_df.to_csv(f"{OUTPUT_DIR}/submission_probabilities.csv", index=False)
