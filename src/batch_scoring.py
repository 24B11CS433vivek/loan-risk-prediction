"""
batch_scoring.py
Batch scoring engine for LoanIQ: read a CSV of applicants, validate it, estimate
blank income / debt-to-income, score every row with one model, build the output.

Pure pandas/numpy (no Streamlit), so it is fast and easy to test.
Whole-column (vectorised) operations are used everywhere, so 5,000 rows take
well under a second to validate and a few seconds at most to score.
"""

import io
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

import predict as _predict
from preprocessing import EDUCATION_CATEGORIES, EMPLOYMENT_CATEGORIES, estimate_dti

# --------------------------------------------------------------------------- schema
TEMPLATE_COLUMNS = [
    "age", "income", "employment_status", "education", "dependents",
    "loan_amount", "loan_term", "credit_score", "existing_debt",
    "debt_to_income_ratio", "previous_defaults",
]
OPTIONAL_COLUMNS = ["income", "debt_to_income_ratio"]          # may be blank: estimated
REQUIRED_COLUMNS = [c for c in TEMPLATE_COLUMNS if c not in OPTIONAL_COLUMNS]
CATEGORICAL = {"employment_status": EMPLOYMENT_CATEGORIES, "education": EDUCATION_CATEGORIES}

LOAN_TERMS = [12, 24, 36, 48, 60, 72, 84]
MAX_ROWS = 100000

# column: (minimum, maximum, whole numbers only)  -- same limits as the Assess page
NUMERIC_RULES = {
    "age":                  (18, 90, True),
    "income":               (1, 100000000, False),
    "dependents":           (0, 10, True),
    "loan_amount":          (1000, 100000000, False),
    "loan_term":            (12, 84, True),
    "credit_score":         (300, 900, True),
    "existing_debt":        (0, 100000000, False),
    "debt_to_income_ratio": (0, 1, False),
    "previous_defaults":    (0, 20, True),
}

RESULT_COLUMNS = ["income_estimated", "dti_estimated", "risk_label", "confidence", "risk_score",
                  "prob_low", "prob_medium", "prob_high", "model"]


class BatchError(Exception):
    """A problem with the file as a whole (shown to the user as a plain message)."""


@dataclass
class Validated:
    raw: pd.DataFrame                 # every uploaded row as text, original index kept
    features: pd.DataFrame            # parsed model columns for every row (NaN where invalid)
    valid: pd.Series                  # True for rows that passed every check
    issues: pd.DataFrame              # row, column, problem, value
    extras: list = field(default_factory=list)   # uploaded columns the model does not use
    missing_optional: list = field(default_factory=list)

    @property
    def n_rows(self):
        return len(self.raw)

    @property
    def n_valid(self):
        return int(self.valid.sum())


# --------------------------------------------------------------------------- template
def template_frame():
    rows = [
        [35, 60000, "Employed",      "Bachelor",    1, 30000, 36, 720,  5000, 0.28, 0],
        [28, None,  "Self-Employed", "Diploma",     0, 45000, 48, 640, 12000, None, 1],
        [52, 95000, "Employed",      "Master",      2, 60000, 60, 780,  8000, 0.22, 0],
        [41, None,  "Part-Time",     "High School", 3, 25000, 24, 590, 15000, 0.55, 2],
    ]
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def template_csv():
    """Bytes of the downloadable template (blank income / DTI cells show the estimator in action)."""
    return template_frame().to_csv(index=False).encode("utf-8")


def column_guide():
    """Rows for the on-screen column reference table."""
    rules = {
        "age": "Whole number, 18 to 90",
        "income": "Annual income in Rs, or blank to estimate",
        "employment_status": ", ".join(EMPLOYMENT_CATEGORIES),
        "education": ", ".join(EDUCATION_CATEGORIES) + " (\"None\" is accepted)",
        "dependents": "Whole number, 0 to 10",
        "loan_amount": "Rs 1,000 to 100,000,000",
        "loan_term": "Months: " + ", ".join(str(t) for t in LOAN_TERMS),
        "credit_score": "Whole number, 300 to 900",
        "existing_debt": "Rs 0 to 100,000,000",
        "debt_to_income_ratio": "0 to 1, or blank to calculate",
        "previous_defaults": "Whole number, 0 to 20",
    }
    return pd.DataFrame([{
        "Column": c, "Required": "No (may be blank)" if c in OPTIONAL_COLUMNS else "Yes", "Accepted values": rules[c],
    } for c in TEMPLATE_COLUMNS])


# --------------------------------------------------------------------------- reading
def _norm_col(name):
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def read_csv_bytes(data):
    """Read uploaded bytes into an all-text DataFrame with normalised column names."""
    if not data or not data.strip():
        raise BatchError("The file is empty.")
    df, last_err = None, None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding=enc,
                             skipinitialspace=True)
            break
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.EmptyDataError:
            raise BatchError("The file has no columns or rows.")
        except pd.errors.ParserError as e:
            raise BatchError("The file could not be read as a comma-separated CSV: %s" % str(e).splitlines()[0])
    if df is None:
        raise BatchError("The file could not be decoded as text: %s" % last_err)

    df.columns = [_norm_col(c) for c in df.columns]
    if len(df.columns) == 1 and re.search(r"[;\t|]", str(df.columns[0])):
        raise BatchError("The file looks semicolon, tab or pipe separated. Save it as a comma-separated CSV.")
    dup = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if dup:
        raise BatchError("Duplicate column names: %s." % ", ".join(dup))
    # drop rows that are completely blank (common trailing rows from spreadsheets)
    blank = df.apply(lambda s: s.str.strip().eq("")).all(axis=1)
    df = df.loc[~blank]
    return df


# --------------------------------------------------------------------------- validation
def _flag(issues, mask, raw_col, column, problem):
    if mask.any():
        idx = mask[mask].index
        issues.append(pd.DataFrame({
            "row": idx.to_numpy() + 2,                      # +2: header is file row 1
            "column": column, "problem": problem, "value": raw_col[mask].to_numpy(),
        }))


def validate(raw):
    """Check columns and values. Returns Validated; raises BatchError for file-level problems."""
    cols = set(raw.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise BatchError(
            "Missing required column(s): %s. Download the template to see the exact column names." % ", ".join(missing))
    if len(raw) == 0:
        raise BatchError("The file has a header but no applicant rows.")
    if len(raw) > MAX_ROWS:
        raise BatchError("The file has %s rows. The limit is %s per upload; split it into smaller files."
                         % (format(len(raw), ","), format(MAX_ROWS, ",")))

    issues, feat = [], pd.DataFrame(index=raw.index)

    for col, (lo, hi, whole) in NUMERIC_RULES.items():
        if col not in cols:
            feat[col] = np.nan                               # optional column left out entirely
            continue
        txt = raw[col].str.strip().str.replace(",", "", regex=False)
        blank = txt.eq("")
        num = pd.to_numeric(txt.where(~blank), errors="coerce")
        num = num.where(np.isfinite(num))
        if col not in OPTIONAL_COLUMNS:
            _flag(issues, blank, raw[col], col, "Value is missing")
        bad_num = ~blank & num.isna()
        _flag(issues, bad_num, raw[col], col, "Not a number")
        ok = num.notna()
        out = ok & ((num < lo) | (num > hi))
        _flag(issues, out, raw[col], col, "Must be between %s and %s" % (format(lo, ","), format(hi, ",")))
        in_range = ok & ~out
        if whole:
            _flag(issues, in_range & (num != num.round()), raw[col], col, "Must be a whole number")
        if col == "loan_term":
            _flag(issues, in_range & ~num.isin(LOAN_TERMS), raw[col], col,
                  "Must be one of " + ", ".join(str(t) for t in LOAN_TERMS))
        feat[col] = num

    for col, allowed in CATEGORICAL.items():
        txt = raw[col].str.strip()
        lookup = {a.lower(): a for a in allowed}
        if col == "education":
            lookup["none"] = "No Formal Education"
        mapped = txt.str.lower().map(lookup)
        blank = txt.eq("")
        _flag(issues, blank, raw[col], col, "Value is missing")
        _flag(issues, ~blank & mapped.isna(), raw[col], col, "Unknown value. Use one of: " + ", ".join(allowed))
        feat[col] = mapped

    issues = (pd.concat(issues, ignore_index=True) if issues
              else pd.DataFrame(columns=["row", "column", "problem", "value"]))
    bad_pos = pd.Index(issues["row"].astype(int).unique() - 2)
    valid = pd.Series(~raw.index.isin(bad_pos), index=raw.index)
    extras = [c for c in raw.columns if c not in TEMPLATE_COLUMNS]
    return Validated(raw=raw, features=feat[TEMPLATE_COLUMNS], valid=valid, issues=issues, extras=extras,
                     missing_optional=[c for c in OPTIONAL_COLUMNS if c not in cols])


def summarize_issues(issues):
    """One line per (column, problem): how many rows, and the first few row numbers."""
    if issues.empty:
        return pd.DataFrame(columns=["Column", "Problem", "Rows affected", "Example rows"])
    g = issues.groupby(["column", "problem"], sort=False)["row"]
    out = pd.DataFrame({
        "Rows affected": g.size(),
        "Example rows": g.apply(lambda s: ", ".join(str(int(x)) for x in s.head(5))),
    }).reset_index().rename(columns={"column": "Column", "problem": "Problem"})
    return out.sort_values("Rows affected", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- estimation
_income_cache = None


def _income_artifact():
    global _income_cache
    if _income_cache is None:
        try:
            _income_cache = joblib.load(_predict.INCOME_MODEL_PATH)
        except FileNotFoundError:
            raise BatchError("Some rows have a blank income, but models/income_estimator.pkl was not found. "
                             "Run  python src/estimators.py  or fill in the income column.")
    return _income_cache


def estimate_incomes(frame):
    """Vectorised version of predict.estimate_income (same model, same rounding to Rs 500)."""
    art = _income_artifact()
    cols = art["features"]
    mu = art["pipeline"].predict(frame[cols])
    return np.maximum(np.round(np.exp(mu) / 500.0) * 500.0, 500.0)


def fill_missing(feat):
    """Estimate blank income first, then blank debt-to-income. Returns (features, income_mask, dti_mask)."""
    feat = feat.copy()
    inc_missing = feat["income"].isna()
    if inc_missing.any():
        feat.loc[inc_missing, "income"] = estimate_incomes(feat.loc[inc_missing])
    dti_missing = feat["debt_to_income_ratio"].isna()
    if dti_missing.any():
        f = feat.loc[dti_missing]
        feat.loc[dti_missing, "debt_to_income_ratio"] = np.round(
            estimate_dti(f["income"], f["existing_debt"], f["loan_amount"], f["loan_term"]), 2)
    return feat, inc_missing, dti_missing


# --------------------------------------------------------------------------- scoring
def score(feat, model_name=None, progress=None, chunk=5000):
    """
    Score every row. Uses the same steps as predict.predict_risk (predict + predict_proba,
    confidence = probability of the predicted class, same risk-score formula) but in
    chunks over whole DataFrames. `progress(fraction)` is called after each chunk.
    """
    artifact = _predict._load_artifact()
    pipeline = _predict._get_pipeline(artifact, model_name)
    classes = [str(c) for c in artifact["label_encoder"].classes_]
    X = feat[artifact["feature_names"]]
    n = len(X)
    codes = np.empty(n, dtype=int)
    proba = np.empty((n, len(classes)), dtype=float)
    for start in range(0, n, chunk):
        part = X.iloc[start:start + chunk]
        codes[start:start + chunk] = np.asarray(pipeline.predict(part), dtype=int)
        proba[start:start + chunk] = pipeline.predict_proba(part)
        if progress:
            progress(min(n, start + chunk) / n)
    ci = {c: i for i, c in enumerate(classes)}
    p4 = np.round(proba, 4)                      # predict_risk also scores from 4-decimal probabilities
    out = pd.DataFrame(index=X.index)
    out["risk_label"] = np.asarray(classes, dtype=object)[codes]
    out["confidence"] = p4[np.arange(n), codes]
    raw_score = 100 * (0.5 * p4[:, ci["Medium"]] + p4[:, ci["High"]])
    out["risk_score"] = [round(float(x), 1) for x in raw_score]       # Python rounding, identical to predict_risk
    for c in ("Low", "Medium", "High"):
        out["prob_" + c.lower()] = p4[:, ci[c]]
    return out


def build_output(validated, feat, inc_missing, dti_missing, result, model_name):
    """Final table: file row, any extra columns, the 11 model inputs (filled in), flags, results."""
    out = pd.DataFrame(index=feat.index)
    out.insert(0, "row", feat.index.to_numpy() + 2)
    taken = set(TEMPLATE_COLUMNS) | set(RESULT_COLUMNS) | {"row"}
    for c in validated.extras:
        out["input_" + c if c in taken else c] = validated.raw.loc[feat.index, c].to_numpy()
    for c in TEMPLATE_COLUMNS:
        out[c] = feat[c].to_numpy()
    for c in ("age", "dependents", "loan_term", "credit_score", "previous_defaults"):
        out[c] = out[c].astype(int)
    for c in ("income", "loan_amount", "existing_debt"):          # whole rupees print as 77500, not 77500.0
        if (out[c] == out[c].round()).all():
            out[c] = out[c].astype(np.int64)
    out["income_estimated"] = np.where(inc_missing.to_numpy(), "Yes", "No")
    out["dti_estimated"] = np.where(dti_missing.to_numpy(), "Yes", "No")
    out = pd.concat([out, result], axis=1)
    out["model"] = model_name
    return out.reset_index(drop=True)
