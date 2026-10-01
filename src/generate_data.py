"""
generate_data.py
Builds a larger, realistic synthetic loan dataset (default 5,000 rows) and
merges it with the original 98 hand-made records.

Usage (from the project root):
    python src/generate_data.py                 # 5000 rows, seed 42
    python src/generate_data.py --rows 8000 --seed 7
"""

import os
import argparse
import numpy as np
import pandas as pd

from preprocessing import (
    EDUCATION_CATEGORIES, EMPLOYMENT_CATEGORIES, estimate_dti
)

BASE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
DATA_PATH = os.path.join(RAW_DIR, "loan_data.csv")
ORIG_PATH = os.path.join(RAW_DIR, "loan_data_original.csv")

TERMS = [12, 24, 36, 48, 60, 72, 84]

# annual income base by education (same money scale as the original dataset)
EDU_BASE_INCOME = {
    "No Formal Education": 14000, "Primary School": 17000, "High School": 24000,
    "Diploma": 32000, "Bachelor": 45000, "Master": 62000, "PhD": 80000,
}
# multiplier on that base by employment type
EMP_INCOME_MULT = {
    "Employed": 1.00, "Self-Employed": 1.05, "Part-Time": 0.45,
    "Retired": 0.55, "Student": 0.25, "Unemployed": 0.35,
}
# extra risk points by employment type
EMP_RISK = {
    "Employed": 0.0, "Self-Employed": 0.25, "Retired": 0.2,
    "Part-Time": 0.6, "Student": 0.8, "Unemployed": 1.5,
}


def _pick(rng, probs):
    names = list(probs.keys())
    p = np.array(list(probs.values()), dtype=float)
    return rng.choice(names, p=p / p.sum())


def _employment(rng, age):
    if age < 25:
        p = {"Student": .35, "Employed": .35, "Part-Time": .12, "Unemployed": .10, "Self-Employed": .08}
    elif age < 55:
        p = {"Employed": .60, "Self-Employed": .17, "Part-Time": .09, "Unemployed": .08, "Student": .06}
    elif age < 60:
        p = {"Employed": .55, "Self-Employed": .17, "Part-Time": .10, "Unemployed": .08, "Retired": .10}
    else:
        p = {"Retired": .55, "Employed": .15, "Self-Employed": .12, "Part-Time": .10, "Unemployed": .08}
    return _pick(rng, p)


def _education(rng, age):
    if age < 25:
        p = {"No Formal Education": .03, "Primary School": .05, "High School": .45,
             "Diploma": .20, "Bachelor": .27}
    elif age < 55:
        p = {"No Formal Education": .04, "Primary School": .07, "High School": .24,
             "Diploma": .13, "Bachelor": .31, "Master": .16, "PhD": .05}
    else:
        p = {"No Formal Education": .10, "Primary School": .15, "High School": .30,
             "Diploma": .10, "Bachelor": .24, "Master": .09, "PhD": .02}
    return _pick(rng, p)


def generate(n_rows=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_rows):
        if rng.random() < 0.12:
            age = int(rng.integers(55, 71))      # older applicants / retirees
        else:
            age = int(np.clip(rng.normal(34, 9), 18, 70))
        emp = _employment(rng, age)
        edu = _education(rng, age)

        # income: education x employment x experience x noise
        experience = 1.0 + 0.015 * np.clip(age - 22, 0, 30)
        sigma = 0.35 if emp == "Self-Employed" else 0.25
        income = EDU_BASE_INCOME[edu] * EMP_INCOME_MULT[emp] * experience * rng.lognormal(0, sigma)
        income = int(round(max(income, 8000) / 500) * 500)

        dependents = int(np.clip(rng.poisson(0.4 + 0.035 * max(age - 22, 0)), 0, 6))

        # requested loan and term
        loan = income * rng.lognormal(np.log(0.55), 0.6)
        loan = int(round(np.clip(loan, 1000, 150000) / 500) * 500)
        idx = int(np.clip(round(loan / 15000) + rng.integers(-1, 2), 0, len(TERMS) - 1))
        term = TERMS[idx]

        # existing debt (a quarter of applicants have none)
        if rng.random() < 0.25 or age < 23:
            debt = 0
        else:
            debt = income * rng.lognormal(np.log(0.18), 0.7)
            debt = int(round(np.clip(debt, 0, 200000) / 500) * 500)

        # behaviour -> previous defaults and credit score
        latent = rng.normal()
        defaults = int(np.clip(rng.poisson(np.exp(-1.7 + 0.8 * latent)), 0, 5))
        score = 690 - 55 * latent - 40 * defaults + 0.8 * (age - 35) + rng.normal(0, 22)
        score = int(np.clip(round(score), 300, 900))

        dti = float(estimate_dti(income, debt, loan, term)) + rng.normal(0, 0.03)
        dti = float(np.clip(round(dti, 2), 0.0, 1.0))

        # hidden risk score -> label (higher = riskier)
        risk = (3.0 * (dti - 0.30)
                + 1.2 * defaults
                - 0.012 * (score - 650)
                + EMP_RISK[emp]
                + 0.6 * (loan / income - 0.6)
                + 0.15 * (dependents - 1)
                - 0.10 * (EDUCATION_CATEGORIES.index(edu) - 3)
                + (0.3 if age < 22 or age > 65 else 0.0)
                + rng.normal(0, 0.35))

        rows.append({
            "age": age, "income": income, "employment_status": emp, "education": edu,
            "dependents": dependents, "loan_amount": loan, "loan_term": term,
            "credit_score": score, "existing_debt": debt, "debt_to_income_ratio": dti,
            "previous_defaults": defaults, "_risk": risk,
        })

    df = pd.DataFrame(rows)
    low_cut, high_cut = df["_risk"].quantile([0.45, 0.75])
    df["risk_label"] = np.where(df["_risk"] < low_cut, "Low",
                         np.where(df["_risk"] < high_cut, "Medium", "High"))
    return df.drop(columns="_risk")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)

    # keep the original 98 records safe, and always merge from that copy
    if not os.path.exists(ORIG_PATH):
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError("data/raw/loan_data.csv not found.")
        pd.read_csv(DATA_PATH).to_csv(ORIG_PATH, index=False)
        print("[OK] original data saved to data/raw/loan_data_original.csv")

    original  = pd.read_csv(ORIG_PATH)
    synthetic = generate(args.rows, args.seed)
    full = pd.concat([original, synthetic], ignore_index=True)
    full = full.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    full.to_csv(DATA_PATH, index=False)

    print(f"[OK] wrote {len(full)} rows ({len(original)} original + {len(synthetic)} generated)")
    print("\nRisk distribution:\n", full["risk_label"].value_counts().to_string())
    print("\nEducation:\n", full["education"].value_counts().to_string())
    print("\nEmployment:\n", full["employment_status"].value_counts().to_string())
    print("\nIncome summary:\n", full["income"].describe().round(0).to_string())


if __name__ == "__main__":
    main()
