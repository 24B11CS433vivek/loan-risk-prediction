"""
predict.py
Inference helper used by the Streamlit app.
Call predict_risk(input_dict) to get risk label + probability + top features.
"""

import os
import joblib
import numpy as np
import pandas as pd

from preprocessing import (
    CATEGORICAL_FEATURES, estimate_dti, monthly_emi,
    EDUCATION_CATEGORIES, EMPLOYMENT_CATEGORIES
)

_HERE         = os.path.dirname(os.path.abspath(__file__))
ALL_MODELS_DIR = os.path.join(_HERE, "..", "models", "all_models")
INCOME_MODEL_PATH = os.path.join(_HERE, "..", "models", "income_estimator.pkl")
DATA_PATH     = os.path.join(_HERE, "..", "data", "raw", "loan_data.csv")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "loan_risk_pipeline.pkl")
_artifact  = None


def _load_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run  python src/train.py  first."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


def predict_risk(input_dict: dict, model_name: str = None) -> dict:
    """
    Parameters
    ----------
    input_dict : dict with keys matching feature names

    Returns
    -------
    dict with keys:
        risk_label      : str  — 'Low' / 'Medium' / 'High'
        probabilities   : dict — {label: probability}
        confidence      : float — probability of predicted class
        top_features    : list of (feature_name, direction) tuples
        message         : str  — plain-English summary
    """
    artifact = _load_artifact()
    pipeline     = _get_pipeline(artifact, model_name)
    le           = artifact["label_encoder"]
    feature_names = artifact["feature_names"]

    # Build input DataFrame in correct column order
    X = pd.DataFrame([input_dict])[feature_names]

    # Predict
    encoded_pred = pipeline.predict(X)[0]
    proba        = pipeline.predict_proba(X)[0]

    risk_label   = le.inverse_transform([encoded_pred])[0]
    classes      = list(le.classes_)
    prob_dict    = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence   = round(float(proba[encoded_pred]), 4)

    # Top contributing features (heuristic from input values)
    contributions = _get_contributions(pipeline, X, feature_names, list(le.classes_))
    top_features  = [(c["feature"], c["direction"]) for c in contributions[:5]]
    risk_score    = round(100 * (0.5 * prob_dict.get("Medium", 0.0)
                                 + 1.0 * prob_dict.get("High", 0.0)), 1)

    # Plain-English message
    messages = {
        "Low":    "The model predicts a lower estimated risk based on the provided applicant information.",
        "Medium": "The model predicts a moderate estimated risk. Some financial indicators need attention.",
        "High":   "The model predicts a higher estimated risk based on the provided applicant information.",
    }

    return {
        "risk_label":    risk_label,
        "probabilities": prob_dict,
        "confidence":    confidence,
        "top_features":  top_features,
        "message":       messages[risk_label],
        "risk_score":    risk_score,
        "contributions": contributions,
        "tips":          make_tips(input_dict),
        "model_used":    model_name or artifact["best_model_name"],
    }


# ── Model selection ──────────────────────────────────────────────────────────
_model_cache = {}

def _get_pipeline(artifact, model_name=None):
    """Tuned best pipeline by default, or any saved model by name."""
    if not model_name or model_name == artifact["best_model_name"]:
        return artifact["pipeline"]
    registry = artifact.get("model_registry", {})
    if model_name not in registry:
        raise ValueError("Unknown model: %s" % model_name)
    if model_name not in _model_cache:
        path = os.path.join(ALL_MODELS_DIR, registry[model_name]["file"])
        _model_cache[model_name] = joblib.load(path)
    return _model_cache[model_name]


def list_models():
    """All trained models with their test metrics, best first."""
    artifact = _load_artifact()
    best = artifact["best_model_name"]
    rows = []
    for name, info in artifact.get("model_registry", {}).items():
        m = info["metrics"]
        rows.append({
            "name": name, "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"], "f1_macro": m["f1_macro"],
            "roc_auc": m["roc_auc"], "tuned": info["tuned"], "selected": name == best,
        })
    return sorted(rows, key=lambda r: r["f1_macro"], reverse=True)


# ── Local explanation (works for every model type) ───────────────────────────
_reference = None

def _reference_row(feature_names):
    """A 'typical applicant': median of numeric columns, mode of categoricals."""
    global _reference
    if _reference is None:
        ref = {}
        if os.path.exists(DATA_PATH):
            d = pd.read_csv(DATA_PATH)
            for c in feature_names:
                ref[c] = d[c].mode().iloc[0] if c in CATEGORICAL_FEATURES else float(d[c].median())
        else:
            ref = {"age": 35.0, "income": 40000.0, "dependents": 1.0, "loan_amount": 20000.0,
                   "loan_term": 36.0, "credit_score": 680.0, "existing_debt": 5000.0,
                   "debt_to_income_ratio": 0.35, "previous_defaults": 0.0,
                   "employment_status": "Employed", "education": "Bachelor"}
        _reference = ref
    return _reference


_RISK_WEIGHT = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
FEATURE_LABELS = {
    "age": "Age", "income": "Annual Income", "dependents": "Dependents",
    "loan_amount": "Loan Amount", "loan_term": "Loan Term",
    "credit_score": "Credit Score", "existing_debt": "Existing Debt",
    "debt_to_income_ratio": "Debt-to-Income Ratio",
    "previous_defaults": "Previous Defaults",
    "employment_status": "Employment Status", "education": "Education",
}

def _get_contributions(pipeline, X, feature_names, classes):
    """
    For each input, swap it for the typical-applicant value and see how the risk
    index (0 = Low, 100 = High) moves. Positive impact = this input raises risk.
    """
    weights = np.array([_RISK_WEIGHT[c] for c in classes])
    base    = float(pipeline.predict_proba(X)[0] @ weights) * 100
    ref     = _reference_row(feature_names)

    variants = pd.concat([X] * len(feature_names), ignore_index=True)
    num_cols = [c for c in feature_names if c not in CATEGORICAL_FEATURES]
    variants[num_cols] = variants[num_cols].astype(float)
    for i, f in enumerate(feature_names):
        variants.at[i, f] = ref[f]

    swapped = pipeline.predict_proba(variants) @ weights * 100
    out = []
    for i, f in enumerate(feature_names):
        impact = round(base - float(swapped[i]), 1)
        value  = X.iloc[0][f]
        if abs(impact) < 0.5:
            direction = "neutral effect"
        elif impact > 0:
            direction = "raises risk"
        else:
            direction = "lowers risk"
        out.append({
            "feature": FEATURE_LABELS.get(f, f), "key": f, "value": value,
            "impact": impact, "direction": direction,
        })
    return sorted(out, key=lambda c: abs(c["impact"]), reverse=True)


# ── Plain-English tips ───────────────────────────────────────────────────────
def make_tips(d):
    tips = []
    income = max(float(d.get("income", 0)), 1.0)
    emi    = float(monthly_emi(d["loan_amount"], d["loan_term"]))
    emi_ratio = emi / (income / 12.0)
    if float(d.get("debt_to_income_ratio", 0)) > 0.40:
        tips.append("Debt-to-income ratio is above 40%. Paying down existing debt before applying would help.")
    if int(d.get("credit_score", 700)) < 650:
        tips.append("A credit score below 650 weighs heavily. Clearing late payments can lift it over time.")
    if int(d.get("previous_defaults", 0)) > 0:
        tips.append("Previous defaults are one of the strongest risk signals in the model.")
    if emi_ratio > 0.35:
        tips.append("The monthly EMI would take over 35% of income. A smaller amount or a longer term lowers it.")
    if float(d["loan_amount"]) > income:
        tips.append("The loan is larger than one year of income. Requesting less would reduce risk.")
    if d.get("employment_status") in ("Unemployed", "Student"):
        tips.append("No stable salary was reported. A co-applicant with steady income would strengthen the case.")
    if not tips:
        tips.append("No major warning signs. Keep credit utilisation low and payments on time.")
    return tips[:4]


# ── Auto-estimators ──────────────────────────────────────────────────────────
_income_artifact = None

def estimate_income(age, education, employment_status, dependents=0):
    """
    Typical annual income for this profile, with a likely range.
    Returns None if models/income_estimator.pkl has not been created yet.
    """
    global _income_artifact
    if _income_artifact is None:
        if not os.path.exists(INCOME_MODEL_PATH):
            return None
        _income_artifact = joblib.load(INCOME_MODEL_PATH)
    row = pd.DataFrame([{
        "age": age, "education": education,
        "employment_status": employment_status, "dependents": dependents,
    }])
    mu    = float(_income_artifact["pipeline"].predict(row)[0])
    sigma = _income_artifact["resid_std"]
    rnd   = lambda v: int(round(v / 500.0) * 500)
    return {
        "estimate": rnd(np.exp(mu)),
        "low":      rnd(np.exp(mu - sigma)),
        "high":     rnd(np.exp(mu + sigma)),
        "r2":       _income_artifact["r2"],
    }


def max_affordable_loan(income, existing_debt, loan_term, target_dti=0.40):
    """Largest loan whose EMI keeps total obligations at or under target_dti."""
    monthly_income = max(float(income), 1.0) / 12.0
    room = target_dti * monthly_income - float(monthly_emi(existing_debt, 36))
    if room <= 0:
        return 0
    unit = float(monthly_emi(1.0, loan_term))      # EMI per 1 unit borrowed
    return int(room / unit // 500 * 500)


def _get_top_features(artifact, X, feature_names):
    """
    Return top 5 features driving the prediction.
    Uses feature_importances_ if available, else falls back to
    domain-knowledge ranking.
    """
    pipeline = artifact["pipeline"]
    clf_step = pipeline.named_steps.get("clf")

    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
    elif hasattr(clf_step, "coef_"):
        importances = np.abs(clf_step.coef_).mean(axis=0)
    else:
        # Fallback: domain-knowledge order
        return [
            ("credit_score", "key factor"),
            ("debt_to_income_ratio", "key factor"),
            ("previous_defaults", "key factor"),
            ("income", "key factor"),
            ("loan_amount", "key factor"),
        ]

    top_idx = np.argsort(importances)[::-1][:5]
    result  = []
    for i in top_idx:
        fname = feature_names[i]
        val   = X.iloc[0][fname]
        # Simple direction label
        direction = _direction_label(fname, val)
        result.append((fname.replace("_", " ").title(), direction))
    return result


def _direction_label(feature, value):
    """Return a human-readable impact label for a feature value."""
    high_risk_high = {"debt_to_income_ratio", "previous_defaults", "existing_debt", "loan_amount"}
    high_risk_low  = {"credit_score", "income"}

    if feature in high_risk_high:
        return "higher value increases risk" if value > 0 else "low value"
    if feature in high_risk_low:
        return "lower value increases risk"
    return "contributing factor"
