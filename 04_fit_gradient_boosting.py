"""Fit and evaluate the leakage-safe HistGradientBoosting model."""
from sklearn.ensemble import HistGradientBoostingClassifier
from utils import fit_evaluate_save, save_model_charts

if __name__ == "__main__":
    metrics = fit_evaluate_save(
        HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=42),
        "Gradient Boosting", "gradient_boosting", "class_weight='balanced' with early_stopping=True",
    )
    save_model_charts(metrics, "gradient_boosting")
    print("Gradient Boosting training complete.")
