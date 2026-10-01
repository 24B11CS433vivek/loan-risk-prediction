from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

NUMERICAL_FEATURES = [
    "age", "income", "dependents",
    "loan_amount", "loan_term", "credit_score",
    "existing_debt", "debt_to_income_ratio", "previous_defaults"
]

CATEGORICAL_FEATURES = ["employment_status", "education"]

# Engineered inside the pipeline from the raw inputs (see add_derived_features)
DERIVED_FEATURES = ["loan_to_income", "monthly_emi", "emi_to_income", "total_debt_burden"]

# Column order the model actually sees (numeric block first, then categorical)
MODEL_FEATURES = NUMERICAL_FEATURES + DERIVED_FEATURES + CATEGORICAL_FEATURES
TARGET = "risk_label"

EDUCATION_CATEGORIES   = ["No Formal Education", "Primary School", "High School",
                          "Diploma", "Bachelor", "Master", "PhD"]
# Label shown in the UI for each stored category (storing the literal string
# "None" is avoided because pandas reads it back as a missing value)
EDUCATION_DISPLAY      = {"No Formal Education": "None"}
EMPLOYMENT_CATEGORIES  = ["Unemployed", "Student", "Retired", "Part-Time",
                          "Self-Employed", "Employed"]

ASSUMED_ANNUAL_RATE    = 0.10   # used for EMI maths
ASSUMED_DEBT_TERM      = 36     # months, existing debt is assumed to run off over this

def monthly_emi(principal, months, annual_rate=ASSUMED_ANNUAL_RATE):
    """Standard EMI formula. Works on scalars or pandas Series / numpy arrays."""
    principal = np.asarray(principal, dtype=float)
    months    = np.maximum(np.asarray(months, dtype=float), 1.0)
    r = annual_rate / 12.0
    growth = (1.0 + r) ** months
    return principal * r * growth / (growth - 1.0)


def estimate_dti(income, existing_debt, loan_amount, loan_term):
    """
    Auto-estimate the debt-to-income ratio:
    (EMI on existing debt + EMI on the new loan) / monthly income, clipped to 0-1.
    """
    monthly_income = np.maximum(np.asarray(income, dtype=float) / 12.0, 1.0)
    obligations = (monthly_emi(existing_debt, ASSUMED_DEBT_TERM)
                   + monthly_emi(loan_amount, loan_term))
    return np.clip(obligations / monthly_income, 0.0, 1.0)


def add_derived_features(df):
    """Adds engineered ratio features. Input and output are DataFrames."""
    df = pd.DataFrame(df).copy()
    income         = np.maximum(df["income"].astype(float), 1.0)
    monthly_income = income / 12.0
    emi            = monthly_emi(df["loan_amount"], df["loan_term"])

    df["loan_to_income"]    = np.clip(df["loan_amount"] / income, 0, 20)
    df["monthly_emi"]       = emi
    df["emi_to_income"]     = np.clip(emi / monthly_income, 0, 5)
    df["total_debt_burden"] = np.clip((df["existing_debt"] + df["loan_amount"]) / income, 0, 30)
    return df


def get_preprocessor():
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[EMPLOYMENT_CATEGORIES, EDUCATION_CATEGORIES],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    column_transformer = ColumnTransformer(transformers=[
        ("num", numerical_pipeline,   NUMERICAL_FEATURES + DERIVED_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    return Pipeline(steps=[
        ("derive", FunctionTransformer(add_derived_features, validate=False)),
        ("columns", column_transformer)
    ])

def validate_dataframe(df):
    required = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    print(f"[OK] Shape: {df.shape}")
    print(f"[OK] Target distribution:\n{df[TARGET].value_counts()}")
    return True
