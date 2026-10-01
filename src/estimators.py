"""
estimators.py
Trains the helper model that auto-estimates a typical annual income from
age, education, employment status and dependents.
Saved to models/income_estimator.pkl and used by the app's "Auto-estimate" button.

Run once after generate_data.py (and again whenever the dataset changes):
    python src/estimators.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from preprocessing import EDUCATION_CATEGORIES, EMPLOYMENT_CATEGORIES

BASE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "raw", "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "income_estimator.pkl")

FEATURES = ["age", "education", "employment_status", "dependents"]
RANDOM_STATE = 42


def build_pipeline():
    pre = ColumnTransformer(transformers=[
        ("num", "passthrough", ["age", "dependents"]),
        ("edu", OrdinalEncoder(categories=[EDUCATION_CATEGORIES],
                               handle_unknown="use_encoded_value", unknown_value=-1),
         ["education"]),
        ("emp", OneHotEncoder(categories=[EMPLOYMENT_CATEGORIES],
                              handle_unknown="ignore"),
         ["employment_status"]),
    ])
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE
    )
    return Pipeline(steps=[("pre", pre), ("reg", reg)])


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], np.log(df["income"].astype(float))

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = build_pipeline().fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    r2   = r2_score(y_te, pred)
    mae  = mean_absolute_error(np.exp(y_te), np.exp(pred))
    resid_std = float(np.std(y_te - pred))
    print(f"[OK] income estimator  R2(log)={r2:.3f}  MAE={mae:,.0f}  resid_std(log)={resid_std:.3f}")

    # refit on all rows for the final saved model
    pipe = build_pipeline().fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipe, "resid_std": resid_std, "r2": float(r2),
                 "mae": float(mae), "features": FEATURES}, MODEL_PATH)
    print("[OK] models/income_estimator.pkl saved")


if __name__ == "__main__":
    main()
