"""Fit and evaluate the leakage-safe XGBoost model."""
import numpy as np
from xgboost import XGBClassifier
from utils import fit_evaluate_save, load_data, save_model_charts

if __name__ == "__main__":
    _, y = load_data()
    negative_count, positive_count = np.bincount(y.astype(int))
    estimator = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        scale_pos_weight=negative_count / positive_count,
        eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist",
    )
    metrics = fit_evaluate_save(estimator, "XGBoost", "xgboost", "scale_pos_weight=negative_count/positive_count")
    save_model_charts(metrics, "xgboost")
    print("XGBoost training complete.")
