"""Fit and evaluate the leakage-safe Random Forest model."""
from sklearn.ensemble import RandomForestClassifier
from utils import fit_evaluate_save, save_model_charts

if __name__ == "__main__":
    metrics = fit_evaluate_save(
        RandomForestClassifier(n_estimators=80, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42),
        "Random Forest", "random_forest", "class_weight='balanced'",
    )
    save_model_charts(metrics, "random_forest")
    print("Random Forest training complete.")
