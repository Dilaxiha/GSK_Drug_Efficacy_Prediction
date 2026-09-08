"""Train the small, 11-raw-feature XGBoost model that app.py actually serves.

This is intentionally separate from the main leakage-safe pipeline (03_fit_xgboost.py etc.),
which trains on ~37 encoder-derived features and isn't what the lightweight Flask API expects.
app.py's FEATURE_COLS is a fixed, smaller clinical feature set, so this script builds exactly
those 11 raw columns from the engineered dataset and fits/saves a single, deployable model to
outputs/xgboost_best_model.pkl.

gender_encoded mapping (clients must send this, not the string): M=1, F=0, Other=2.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from utils import DATA_PATH, RANDOM_STATE

FEATURE_COLS = [
    "age", "gender_encoded", "bmi", "dosage_mg",
    "hemoglobin", "creatinine", "egfr", "hba1c",
    "concurrent_drugs", "liver_risk", "polypharmacy",
]
GENDER_MAP = {"M": 1, "F": 0, "Other": 2}
OUTPUT_PATH = Path(__file__).resolve().parent / "outputs" / "xgboost_best_model.pkl"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["age"] = df["age"]
    features["gender_encoded"] = df["gender"].map(GENDER_MAP)
    features["bmi"] = df["bmi"]
    features["dosage_mg"] = df["dosage"].clip(lower=0)  # raw data has a few negative glitches
    features["hemoglobin"] = df["hemoglobin"]
    features["creatinine"] = df["creatinine"]
    features["egfr"] = df["egfr"]
    features["hba1c"] = df["hba1c"]
    features["concurrent_drugs"] = df["concurrent_drugs"]
    features["liver_risk"] = df["liver_risk"]
    features["polypharmacy"] = df["polypharmacy"]
    return features[FEATURE_COLS]


def main() -> None:
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    X = build_features(df).astype("float32")
    y = df["treatment_outcome"].astype("int8")
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid]
    print(f"Training rows after dropping incomplete cases: {len(X):,} (dropped {(~valid).sum():,})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    negative_count, positive_count = np.bincount(y_train)
    scale_pos_weight = float(negative_count / positive_count)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        n_jobs=2, random_state=RANDOM_STATE, tree_method="hist",
    )
    print("Fitting XGBoost on the 11 deployment features...")
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]
    print(f"Test PR-AUC: {average_precision_score(y_test, test_probs):.4f}")
    print(classification_report(y_test, (test_probs >= 0.5).astype(int), target_names=["Ineffective", "Effective"], zero_division=0))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")
    print(f"Feature order: {FEATURE_COLS}")
    print(f"gender_encoded mapping: {GENDER_MAP}")


if __name__ == "__main__":
    main()
