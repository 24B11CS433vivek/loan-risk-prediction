"""
evaluate.py
Loads the saved pipeline and prints a full evaluation report.
Run this any time after train.py to re-check metrics without retraining.
"""

import os
import joblib

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from any folder
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from preprocessing import (
    validate_dataframe, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, TARGET
)

RANDOM_STATE = 42
FIGURES_DIR  = os.path.join("..", "reports", "figures")
MODEL_PATH   = os.path.join("..", "models", "loan_risk_pipeline.pkl")

# ── 1. Load artifact ─────────────────────────────────────────────────────────
print("\n=== Loading saved pipeline ===")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Pipeline not found. Run src/train.py first.")

artifact = joblib.load(MODEL_PATH)
pipeline       = artifact["pipeline"]
le             = artifact["label_encoder"]
best_model     = artifact["best_model_name"]
test_metrics   = artifact["test_metrics"]
comparison_df  = artifact["comparison_df"]

print(f"[OK] Loaded pipeline — best model: {best_model}")

# ── 2. Reload data and recreate test split (same seed) ───────────────────────
df = pd.read_csv(os.path.join("..", "data", "raw", "loan_data.csv"))
validate_dataframe(df)

df["risk_encoded"] = le.transform(df[TARGET])
X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df["risk_encoded"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

# ── 3. Print metrics ─────────────────────────────────────────────────────────
print("\n=== Test Set Metrics ===")
print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
print(f"  Precision : {test_metrics['precision']:.4f}")
print(f"  Recall    : {test_metrics['recall']:.4f}")
print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== Model Comparison Table ===")
print(comparison_df.to_string(index=False))

# ── 4. ROC curves (one-vs-rest) ──────────────────────────────────────────────
print("\n=== Generating ROC curves ===")
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, class_name in enumerate(le.classes_):
    ax = axes[i]
    RocCurveDisplay.from_predictions(
        y_test_bin[:, i],
        y_prob[:, i],
        name=f"{class_name} vs Rest",
        ax=ax,
        color="#1e3a5f"
    )
    ax.set_title(f"ROC Curve — {class_name}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

plt.suptitle("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
plt.close()
print("[OK] roc_curves.png saved")

# ── 5. Feature importance (if supported) ─────────────────────────────────────
print("\n=== Feature Importance ===")
clf_step = pipeline.named_steps.get("clf")
feature_names = MODEL_FEATURES

if hasattr(clf_step, "feature_importances_"):
    importances = clf_step.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#2e86ab")
    ax.set_title(f"Feature Importance — {best_model}")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("[OK] feature_importance.png saved")
    print(fi_df.sort_values("Importance", ascending=False).to_string(index=False))
elif hasattr(clf_step, "coef_"):
    coef = np.abs(clf_step.coef_).mean(axis=0)
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": coef
    }).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["Feature"], fi_df["Importance"], color="#2e86ab")
    ax.set_title(f"Coefficient Magnitude — {best_model}")
    ax.set_xlabel("|Coefficient|")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("[OK] feature_importance.png saved (coefficients)")
else:
    print("[INFO] This model does not expose feature importances directly.")

print("\n=== EVALUATION COMPLETE ===")
