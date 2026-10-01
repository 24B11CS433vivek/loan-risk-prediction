"""
consensus.py - logic for the "Consensus" section on the Assess Applicant page (no Streamlit imports).

Runs one applicant through every saved model, takes a majority vote and decides
whether the models disagree enough to recommend manual review.
"""
import numpy as np
import pandas as pd

from predict import _load_artifact, _get_pipeline

AGREEMENT_THRESHOLD = 0.70   # below this share of votes, flag for manual review
SPLIT_MARGIN = 1             # top two levels within this many votes = split
ORDER = ["Low", "Medium", "High"]


def run_all(applicant):
    """
    Score `applicant` (dict with the 11 model input fields) with every saved model.
    Returns (table, failed): table has one row per model with columns
    model, label, confidence, risk_score, Low, Medium, High; failed lists models that could not run.
    """
    art = _load_artifact()
    feats = art["feature_names"]
    X = pd.DataFrame([applicant])[feats]
    classes = list(art["label_encoder"].classes_)
    rows, failed = [], []
    for name in art.get("model_registry", {}):
        try:
            pipe = _get_pipeline(art, name)
            p = pipe.predict_proba(X)[0]
        except Exception as e:                       # a missing or incompatible model file must not kill the page
            failed.append("%s (%s)" % (name, type(e).__name__))
            continue
        probs = {c: float(v) for c, v in zip(classes, p)}
        label = classes[int(np.argmax(p))]
        rows.append({
            "model": name, "label": label, "confidence": probs[label],
            "risk_score": round(100 * (0.5 * probs.get("Medium", 0.0) + probs.get("High", 0.0)), 1),
            **{c: probs.get(c, 0.0) for c in ORDER},
        })
    return pd.DataFrame(rows), failed


def summarize(table):
    """Majority vote, agreement share and the manual-review decision."""
    n = len(table)
    counts = table["label"].value_counts()
    votes = {k: int(counts.get(k, 0)) for k in ORDER}
    top = max(votes.values())
    leaders = [k for k in ORDER if votes[k] == top]
    if len(leaders) > 1:                              # tie: break it with the average probability
        mean_p = {k: float(table[k].mean()) for k in leaders}
        majority = max(mean_p, key=mean_p.get)
    else:
        majority = leaders[0]
    agreement = votes[majority] / n
    ranked = sorted(votes.values(), reverse=True)
    split = len(ranked) > 1 and ranked[1] > 0 and (ranked[0] - ranked[1]) <= SPLIT_MARGIN
    reasons = []
    if agreement < AGREEMENT_THRESHOLD:
        reasons.append("only %.0f%% of models agree" % (agreement * 100))
    if split:
        reasons.append("the top two risk levels are split (%d vs %d votes)" % (ranked[0], ranked[1]))
    return {
        "n": n, "votes": votes, "majority": majority, "agreement": agreement,
        "disagree_models": table.loc[table["label"] != majority, "model"].tolist(),
        "review": bool(reasons), "reasons": reasons,
        "mean_prob": {k: float(table[k].mean()) for k in ORDER},
        "mean_score": float(table["risk_score"].mean()),
    }
