"""Fit and evaluate the leakage-safe Decision Tree model."""
from sklearn.tree import DecisionTreeClassifier
from utils import fit_evaluate_save, save_model_charts

if __name__ == "__main__":
    metrics = fit_evaluate_save(
        DecisionTreeClassifier(max_depth=7,min_samples_leaf=50,min_samples_split=100, class_weight="balanced", random_state=42),
        "Decision Tree", "decision_tree", "class_weight='balanced'",
    )
    save_model_charts(metrics, "decision_tree")
    print("Decision Tree training complete.")
