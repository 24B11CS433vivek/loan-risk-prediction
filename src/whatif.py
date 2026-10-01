"""
whatif.py - logic for the What-If Simulator page (no Streamlit imports).

Scores many hypothetical applicants in one vectorised call and searches for the
smallest realistic changes that flip the prediction to Low risk.
"""
import itertools

import numpy as np
import pandas as pd

from predict import _load_artifact, _get_pipeline
from preprocessing import estimate_dti, monthly_emi

TERMS = [12, 24, 36, 48, 60, 72, 84]

# Applicant used when no assessment has been run yet (same defaults as Assess Applicant)
DEFAULT_BASELINE = {
    "age": 35, "employment_status": "Employed", "education": "Bachelor", "dependents": 1,
    "income": 60000, "loan_amount": 30000, "loan_term": 36, "credit_score": 700,
    "existing_debt": 5000, "previous_defaults": 0,
}

# Fields the simulator can move
LEVERS = ["credit_score", "loan_amount", "loan_term", "existing_debt"]
LEVER_LABEL = {"credit_score": "Credit score", "loan_amount": "Loan amount",
               "loan_term": "Loan term", "existing_debt": "Existing debt"}


def normalise_baseline(src):
    """Build a clean baseline dict from a last-assessment payload (or None)."""
    base = dict(DEFAULT_BASELINE)
    if src:
        for k in base:
            if k in src and src[k] is not None:
                base[k] = src[k]
    base["loan_term"] = min(TERMS, key=lambda t: abs(t - int(base["loan_term"])))
    for k in ("age", "dependents", "credit_score", "previous_defaults"):
        base[k] = int(base[k])
    for k in ("income", "loan_amount", "existing_debt"):
        base[k] = float(base[k])
    return base


def _frame(rows):
    """Candidate dicts -> model input frame (debt-to-income recalculated for each row)."""
    df = pd.DataFrame(rows)
    df["debt_to_income_ratio"] = np.round(estimate_dti(
        df["income"].values, df["existing_debt"].values, df["loan_amount"].values, df["loan_term"].values), 2)
    return df


def score_frame(df, model_name=None):
    """Return a frame with class probabilities, risk_score and predicted label for every row."""
    art = _load_artifact()
    pipe = _get_pipeline(art, model_name)
    feats = art["feature_names"]
    X = df[feats]
    proba = pipe.predict_proba(X)
    classes = list(art["label_encoder"].classes_)
    out = pd.DataFrame(proba, columns=classes, index=df.index)
    out["risk_score"] = np.round(100 * (0.5 * out.get("Medium", 0.0) + 1.0 * out.get("High", 0.0)), 1)
    out["label"] = [classes[i] for i in proba.argmax(axis=1)]
    return out


def score_one(applicant, model_name=None):
    """Score one applicant dict. Returns dict with label, risk_score, probabilities, emi, dti."""
    df = _frame([applicant])
    s = score_frame(df, model_name).iloc[0]
    classes = list(_load_artifact()["label_encoder"].classes_)
    return {
        "label": s["label"], "risk_score": float(s["risk_score"]),
        "probabilities": {c: float(s[c]) for c in classes},
        "emi": float(monthly_emi(applicant["loan_amount"], applicant["loan_term"])),
        "dti": float(df["debt_to_income_ratio"].iloc[0]),
    }


def _grid(applicant):
    """Realistic candidate values for each lever (never worse than the current value)."""
    cs = int(applicant["credit_score"])
    credit = sorted({min(900, cs + d) for d in range(0, 151, 10)})
    loan0 = float(applicant["loan_amount"])
    loans = sorted({max(1000.0, round(loan0 * f / 500) * 500) for f in np.arange(1.0, 0.29, -0.05)} | {loan0})
    loans = [x for x in loans if x <= loan0]
    terms = [t for t in TERMS if t >= int(applicant["loan_term"])]
    debt0 = float(applicant["existing_debt"])
    debts = sorted({round(debt0 * f / 500) * 500 for f in np.arange(1.0, -0.01, -0.125)} | {debt0})
    debts = [x for x in debts if 0 <= x <= debt0]
    return {"credit_score": credit, "loan_amount": loans, "loan_term": terms, "existing_debt": debts}


def _cost(applicant, c):
    """Effort of a change: each lever scaled to roughly 0-1, then summed."""
    cost = (c["credit_score"] - applicant["credit_score"]) / 100.0
    if applicant["loan_amount"] > 0:
        cost += (applicant["loan_amount"] - c["loan_amount"]) / applicant["loan_amount"]
    if applicant["existing_debt"] > 0:
        cost += (applicant["existing_debt"] - c["existing_debt"]) / applicant["existing_debt"]
    cost += (c["loan_term"] - applicant["loan_term"]) / 48.0
    return cost


def find_low_risk(applicant, model_name=None, max_results=5):
    """
    Search credit score, loan amount, loan term and existing debt for the smallest
    changes that make the model predict Low. Returns (rows, status) where status is
    'already_low', 'found' or 'none'. Each row: changes (list of (lever, old, new)),
    risk_score, emi, cost.
    """
    base = score_one(applicant, model_name)
    if base["label"] == "Low":
        return [], "already_low"

    g = _grid(applicant)
    combos = list(itertools.product(g["credit_score"], g["loan_amount"], g["loan_term"], g["existing_debt"]))
    rows = []
    for cs, la, lt, ed in combos:
        r = dict(applicant)
        r.update(credit_score=cs, loan_amount=la, loan_term=lt, existing_debt=ed)
        rows.append(r)
    df = _frame(rows)
    sc = score_frame(df, model_name)
    hit = sc["label"].values == "Low"
    if not hit.any():
        return [], "none"

    cands = []
    for i in np.flatnonzero(hit):
        c = rows[i]
        cands.append({"cand": c, "cost": _cost(applicant, c), "risk_score": float(sc["risk_score"].iloc[i])})
    cands.sort(key=lambda x: (x["cost"], x["risk_score"]))

    def changed(c):
        return {k for k in LEVERS if c[k] != applicant[k]}

    kept = []
    for x in cands:
        ch = changed(x["cand"])
        # skip if a cheaper kept option already makes a subset of these changes
        # at the same or smaller size on every lever
        dominated = False
        for y in kept:
            yc = y["cand"]
            if all(abs(yc[k] - applicant[k]) <= abs(x["cand"][k] - applicant[k]) for k in LEVERS):
                dominated = True
                break
        if dominated or not ch:
            continue
        kept.append(x)
        if len(kept) >= max_results:
            break

    out = []
    for x in kept:
        c = x["cand"]
        out.append({
            "changes": [(k, applicant[k], c[k]) for k in LEVERS if c[k] != applicant[k]],
            "risk_score": x["risk_score"],
            "emi": float(monthly_emi(c["loan_amount"], c["loan_term"])),
            "cost": x["cost"],
        })
    return out, "found"
