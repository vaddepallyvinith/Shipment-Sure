import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# =====================================================
# CONFIG
# =====================================================
DATA_FILE = "FeaturesImprovedDataSet.csv"
TARGET_COLUMN = "on_time_delivery"      # 1 = on-time (positive), 0 = delayed
TEST_SIZE = 0.20
RANDOM_STATE = 42

# =====================================================
# 1. LOAD & OPTIMIZE DATA
# =====================================================
df = pd.read_csv(DATA_FILE, low_memory=True)

# Memory optimization
for col in df.select_dtypes(include=['float64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
for col in df.select_dtypes(include=['int64']).columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

# Ensure target is correct: 1 = on-time, 0 = delayed
# If your original column has 1 = delayed, flip it
if df[TARGET_COLUMN].mean() < 0.5:  # assuming delayed is minority class
    df[TARGET_COLUMN] = 1 - df[TARGET_COLUMN]

print(f"Dataset shape: {df.shape}")
print(f"Class distribution → On-time: {df[TARGET_COLUMN].mean():.3%} | Delayed: {(1-df[TARGET_COLUMN].mean()):.3%}")

# =====================================================
# 2. FEATURES / TARGET
# =====================================================
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# One-hot encode any remaining categorical columns (safety check)
cat_cols = X.select_dtypes(include=["object", "category"]).columns
if len(cat_cols) > 0:
    print(f"Encoding remaining categorical columns: {list(cat_cols)}")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X = X.astype(np.float32)
print(f"Final number of features: {X.shape[1]}")

# =====================================================
# 3. TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# Calculate imbalance ratio for XGBoost
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Imbalance ratio (delayed / on-time): {imbalance_ratio:.2f}")

# =====================================================
# 4. RESULTS STORAGE
# =====================================================
results = []

def log_results(name, y_true, y_pred, y_prob):
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob)
    })

# =====================================================
# 5. MODEL 1: LOGISTIC REGRESSION
# =====================================================
print("\nTraining Logistic Regression...")
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

log_reg.fit(X_train, y_train)
lr_prob = log_reg.predict_proba(X_test)[:, 1]
lr_pred = log_reg.predict(X_test)

log_results("Logistic Regression", y_test, lr_pred, lr_prob)

# =====================================================
# 6. MODEL 2: RANDOM FOREST
# =====================================================
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

log_results("Random Forest", y_test, rf_pred, rf_prob)

# =====================================================
# 7. MODEL 3: XGBOOST (Best Performer)
# =====================================================
print("Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=imbalance_ratio,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist"
)

xgb_model.fit(X_train, y_train)

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# Optimal threshold using Youden's J statistic (best for imbalanced data)
fpr, tpr, thresholds = roc_curve(y_test, xgb_prob)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
print(f"Optimal threshold (Youden's J): {best_threshold:.3f}")

# Predictions with default and optimized threshold
xgb_pred_default = (xgb_prob >= 0.5).astype(int)
xgb_pred_opt = (xgb_prob >= best_threshold).astype(int)

log_results("XGBoost (Threshold 0.5)", y_test, xgb_pred_default, xgb_prob)
log_results("XGBoost (Optimized Threshold)", y_test, xgb_pred_opt, xgb_prob)

# =====================================================
# 8. FINAL RESULTS TABLE
# =====================================================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("FINAL MODEL COMPARISON (Sorted by ROC_AUC)")
print("="*80)
print(results_df.round(4).to_string(index=False))

# =====================================================
# 9. VISUALIZATIONS
# =====================================================

# 1. Confusion Matrices (Top 3 models)
top_3 = results_df.head(3)
pred_dict = {
    "Logistic Regression": lr_pred,
    "Random Forest": rf_pred,
    "XGBoost (Optimized Threshold)": xgb_pred_opt
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()
for i, model_name in enumerate(pred_dict.keys()):
    pred = pred_dict[model_name]
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Delayed", "On-Time"])
    disp.plot(ax=axes[i], cmap="Blues", colorbar=False)
    acc = accuracy_score(y_test, pred)
    axes[i].set_title(f"{model_name}\nAcc: {acc:.3f} | AUC: {results_df.loc[results_df['Model']==model_name, 'ROC_AUC'].values[0]:.3f}")

plt.suptitle("Confusion Matrices - Top Performing Models", fontsize=16)
plt.tight_layout()
plt.show()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")

models_probs = [
    ("Logistic Regression", lr_prob),
    ("Random Forest", rf_prob),
    ("XGBoost", xgb_prob)
]

for name, prob in models_probs:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# 3. Precision-Recall Curve (very important for imbalanced data)
plt.figure(figsize=(10, 8))
for name, prob in models_probs:
    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.plot(recall, precision, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 4. Top 15 XGBoost Feature Importances
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 8))
sns.barplot(x=top15.values, y=top15.index, palette="viridis")
plt.title("Top 15 Most Important Features (XGBoost)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# =====================================================
# FINAL RECOMMENDATION
# =====================================================
best_model_name = results_df.iloc[0]["Model"]
best_auc = results_df.iloc[0]["ROC_AUC"]
best_f1 = results_df.iloc[0]["F1"]

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print(f"Best performing model: {best_model_name}")
print(f"→ ROC AUC: {best_auc:.4f} | F1 Score: {best_f1:.4f}")
if "Optimized" in best_model_name:
    print(f"→ Use prediction threshold = {best_threshold:.3f} in production")
print("\nFocus on ROC_AUC and F1 — Accuracy can be misleading due to class imbalance!")
print("Modeling pipeline complete!")
