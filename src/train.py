"""
train.py
Loads data, runs EDA, trains and compares 4 models,
tunes best model, evaluates on test set, saves pipeline.
"""

import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from any folder
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, cross_validate,
    StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from sklearn.preprocessing import label_binarize, LabelEncoder

from preprocessing import (
    get_preprocessor, validate_dataframe,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, TARGET
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FIGURES_DIR = os.path.join("..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. Load & validate ───────────────────────────────────────────────────────
print("\n=== STAGE: Load & Validate ===")
df = pd.read_csv(os.path.join("..", "data", "raw", "loan_data.csv"))
validate_dataframe(df)
print(df.dtypes)
print(df.describe())

# ── 2. EDA ───────────────────────────────────────────────────────────────────
print("\n=== STAGE: EDA ===")

# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
df[TARGET].value_counts().plot(kind="bar", ax=ax, color=["#1e3a5f", "#2e86ab", "#e84855"])
ax.set_title("Loan Risk Distribution")
ax.set_xlabel("Risk Level")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "target_distribution.png"), dpi=150)
plt.close()
print("[OK] target_distribution.png saved")

# Correlation heatmap (numerical only)
fig, ax = plt.subplots(figsize=(10, 7))
corr = df[NUMERICAL_FEATURES].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print("[OK] correlation_heatmap.png saved")

# Credit score vs Risk
fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column="credit_score", by=TARGET, ax=ax)
ax.set_title("Credit Score by Risk Level")
ax.set_xlabel("Risk Level")
ax.set_ylabel("Credit Score")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "credit_score_by_risk.png"), dpi=150)
plt.close()
print("[OK] credit_score_by_risk.png saved")

# ── 3. Encode target ─────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(["Low", "Medium", "High"])
df["risk_encoded"] = le.transform(df[TARGET])
print(f"\nLabel classes: {list(le.classes_)}")

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df["risk_encoded"]

# ── 4. Stratified train/test split ───────────────────────────────────────────
print("\n=== STAGE: Train/Test Split (80/20 stratified) ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ── 5. Define models ─────────────────────────────────────────────────────────
preprocessor = get_preprocessor()

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, algorithm="SAMME", random_state=RANDOM_STATE
    ),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1, verbosity=0
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
    "Support Vector Machine": SVC(
        probability=True, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True,
        random_state=RANDOM_STATE
    ),
    "Naive Bayes": GaussianNB(),
}

# ── 6. Cross-validation comparison ───────────────────────────────────────────
print("\n=== STAGE: Cross-Validation (5-fold) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, clf in models.items():
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "f1_macro", "roc_auc_ovr"],
        return_train_score=True, n_jobs=-1
    )
    results.append({
        "Model": name,
        "CV Accuracy": round(scores["test_accuracy"].mean(), 4),
        "CV F1 (macro)": round(scores["test_f1_macro"].mean(), 4),
        "CV ROC-AUC": round(scores["test_roc_auc_ovr"].mean(), 4),
        "Train Accuracy": round(scores["train_accuracy"].mean(), 4),
        "Fit Time (s)": round(scores["fit_time"].mean(), 2),
    })
    print(f"  {name}: acc={results[-1]['CV Accuracy']}  f1={results[-1]['CV F1 (macro)']}  auc={results[-1]['CV ROC-AUC']}")

comparison_df = pd.DataFrame(results).sort_values("CV F1 (macro)", ascending=False)
print("\nModel Comparison Table:")
print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join("..", "reports", "model_comparison.csv"), index=False)

# ── 7. Select best model by CV F1 ────────────────────────────────────────────
best_model_name = comparison_df.iloc[0]["Model"]
print(f"\n[SELECTED] Best model: {best_model_name}")

# ── 8. Hyperparameter tuning on best model ───────────────────────────────────
print("\n=== STAGE: Hyperparameter Tuning ===")

param_grids = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["lbfgs", "saga"]
    },
    "Decision Tree": {
        "clf__max_depth": [3, 5, 7, None],
        "clf__min_samples_split": [2, 5, 10]
    },
    "Random Forest": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, None]
    },
    "Gradient Boosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [3, 5]
    },
    "Extra Trees": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20]
    },
    "AdaBoost": {
        "clf__n_estimators": [50, 100, 200],
        "clf__learning_rate": [0.5, 1.0]
    },
    "Hist Gradient Boosting": {
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_iter": [100, 200],
        "clf__max_depth": [None, 4, 8]
    },
    "XGBoost": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.05, 0.1]
    },
    "K-Nearest Neighbors": {
        "clf__n_neighbors": [5, 11, 15, 25],
        "clf__weights": ["uniform", "distance"]
    },
    "Support Vector Machine": {
        "clf__C": [0.5, 1, 5],
        "clf__gamma": ["scale", "auto"]
    },
    "Neural Network (MLP)": {
        "clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
        "clf__alpha": [0.0001, 0.001]
    },
    "Naive Bayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7]
    },
}

best_clf = models[best_model_name]
best_pipe = Pipeline(steps=[("pre", get_preprocessor()), ("clf", best_clf)])
grid = GridSearchCV(
    best_pipe,
    param_grids[best_model_name],
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV F1 (macro): {round(grid.best_score_, 4)}")
final_pipeline = grid.best_estimator_

# ── 9. Final evaluation on test set ─────────────────────────────────────────
print("\n=== STAGE: Test Set Evaluation ===")
y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 (macro): {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("[OK] confusion_matrix.png saved")

# Model comparison bar chart
fig, ax = plt.subplots(figsize=(14, 5.5))
x = np.arange(len(comparison_df))
width = 0.25
ax.bar(x - width, comparison_df["CV Accuracy"], width, label="Accuracy", color="#1e3a5f")
ax.bar(x,         comparison_df["CV F1 (macro)"], width, label="F1 (macro)", color="#2e86ab")
ax.bar(x + width, comparison_df["CV ROC-AUC"], width, label="ROC-AUC", color="#e84855")
ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"], rotation=30, ha="right")
ax.set_ylim(0, 1.1)
ax.set_title("Model Comparison (CV Scores)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
plt.close()
print("[OK] model_comparison.png saved")

# ── 9b. Evaluate every model on the test set and save each one ──────────────
print("\n=== STAGE: Model Registry (all models) ===")
ALL_MODELS_DIR = os.path.join("..", "models", "all_models")
os.makedirs(ALL_MODELS_DIR, exist_ok=True)

def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

model_registry     = {}
confusion_matrices = {}

for name, clf in models.items():
    if name == best_model_name:
        pipe = final_pipeline                      # tuned version
    else:
        pipe = Pipeline(steps=[("pre", get_preprocessor()), ("clf", clone(clf))])
        pipe.fit(X_train, y_train)

    p_pred = pipe.predict(X_test)
    p_prob = pipe.predict_proba(X_test)
    m = {
        "accuracy":  accuracy_score(y_test, p_pred),
        "precision": precision_score(y_test, p_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_test, p_pred, average="macro", zero_division=0),
        "f1_macro":  f1_score(y_test, p_pred, average="macro", zero_division=0),
        "roc_auc":   roc_auc_score(y_test_bin, p_prob, multi_class="ovr", average="macro"),
    }
    fname = _slug(name) + ".pkl"
    joblib.dump(pipe, os.path.join(ALL_MODELS_DIR, fname))
    model_registry[name] = {
        "file": fname, "metrics": m, "tuned": name == best_model_name
    }
    confusion_matrices[name] = confusion_matrix(y_test, p_pred).tolist()
    print(f"  {name:<26} acc={m['accuracy']:.3f}  f1={m['f1_macro']:.3f}  auc={m['roc_auc']:.3f}"
          + ("  [tuned, selected]" if name == best_model_name else ""))

# ROC curves of the selected model (resampled to 101 points for plotting)
fpr_grid = np.linspace(0, 1, 101)
roc_data = {}
for i, cname in enumerate(le.classes_):
    fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_data[cname] = {
        "fpr": fpr_grid.tolist(),
        "tpr": np.interp(fpr_grid, fpr_i, tpr_i).tolist(),
        "auc": float(roc_auc_score(y_test_bin[:, i], y_prob[:, i])),
    }

# Permutation importance on the RAW input columns (works for any model type)
perm = permutation_importance(
    final_pipeline, X_test, y_test, scoring="f1_macro",
    n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature":    list(X_test.columns),
    "importance": perm.importances_mean,
    "std":        perm.importances_std,
}).sort_values("importance", ascending=False).reset_index(drop=True)
print("\nTop features (permutation importance):")
print(perm_df.head(6).to_string(index=False))

# ── 10. Save pipeline + metadata ─────────────────────────────────────────────
print("\n=== STAGE: Save Pipeline ===")
artifact = {
    "pipeline": final_pipeline,
    "label_encoder": le,
    "best_model_name": best_model_name,
    "feature_names": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    "input_features": NUMERICAL_FEATURES + CATEGORICAL_FEATURES,
    "model_features": MODEL_FEATURES,
    "class_names": list(le.classes_),
    "model_registry": model_registry,
    "confusion_matrices": confusion_matrices,
    "roc_data": roc_data,
    "permutation_importance": perm_df,
    "dataset_info": {
        "rows": int(len(df)), "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "class_counts": {k: int(v) for k, v in df[TARGET].value_counts().items()},
        "best_params": grid.best_params_,
    },
    "test_metrics": {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1_macro": f1, "roc_auc": auc
    },
    "comparison_df": comparison_df
}
joblib.dump(artifact, os.path.join("..", "models", "loan_risk_pipeline.pkl"))
print("[OK] models/loan_risk_pipeline.pkl saved")
print("\n=== TRAINING COMPLETE ===")
