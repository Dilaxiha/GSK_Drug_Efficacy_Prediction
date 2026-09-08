"""Model interpretability for the tuned XGBoost model: global/local SHAP + a LIME sanity check.

Clinician-facing framing used throughout:
- SHAP/LIME "push" a prediction away from a baseline (average) rate toward this patient's rate.
- Red / positive weight  -> pushes the predicted probability UP (toward "Effective").
- Blue / negative weight -> pushes the predicted probability DOWN (toward "Ineffective").
- Bar/point magnitude    -> how much that feature moved the prediction for this patient.
"""
import joblib
import matplotlib
matplotlib.use("Agg")  # headless (no display) -> figures are written straight to disk
import matplotlib.pyplot as plt
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

from utils import METRICS_DIR, ROOT, get_train_val_test_data

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load the tuned model bundle + the exact same leakage-safe split used to train it ---
bundle = joblib.load(METRICS_DIR / "xgboost.pkl")
best_xgb = bundle["estimator"]
X_train, _, X_test, y_train, _, y_test = get_train_val_test_data()

y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

# =====================================================================================
# 1) SHAP: global summary (population-level feature impact) + local force plot (patient 0)
# =====================================================================================
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer(X_test)

# --- Global summary plot ---
# Each dot is one patient. Position on the x-axis = that feature's push on THIS patient's
# prediction (left = toward "Ineffective", right = toward "Effective"). Color = the patient's
# own value for that feature (red = high, blue = low), so clinicians can read both direction
# and "is it a high or low value doing the pushing" at a glance. Features are ranked top-to-
# bottom by overall importance (average magnitude of impact across all patients).
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Local force plot for a single patient ---
patient_idx = 0
print(f"Patient {patient_idx} prediction: {y_prob_xgb[patient_idx]:.3f}")
print(f"Actual outcome: {y_test.iloc[patient_idx]}")

# Safely handle expected value (binary classification case)
base_value = (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

# Safely handle SHAP values shape for binary classification: depending on SHAP version this is
# either a list of per-class arrays, a 3-D array (patients x features x classes), or already
# reduced to a single 2-D array of "push toward class 1" values.
raw_values = shap_values.values if hasattr(shap_values, "values") else shap_values
if isinstance(raw_values, list):
    patient_shap = raw_values[1][patient_idx]
elif np.ndim(raw_values) == 3:
    patient_shap = raw_values[patient_idx, :, 1]
else:
    patient_shap = raw_values[patient_idx]

# Clear any active figure prior to rendering
plt.figure()

# Generate Force Plot: red bars push this patient's prediction toward "Effective"; blue bars
# push it toward "Ineffective". Bar width = magnitude of that feature's contribution.
shap.force_plot(
    base_value,
    patient_shap,
    X_test.iloc[patient_idx],
    feature_names=X_test.columns,
    matplotlib=True,
    show=False,
)

# Save and release figure memory
plt.savefig(OUTPUT_DIR / "shap_force_patient0.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================================================
# 2) LIME: independent local explanation for the same patient, as a cross-check on SHAP
# =====================================================================================
lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=list(X_train.columns),
    class_names=["Ineffective", "Effective"],
    mode="classification",
    random_state=42,
)

lime_explanation = lime_explainer.explain_instance(
    X_test.iloc[0].values,
    best_xgb.predict_proba,
    num_features=10,
)

# Positive weight -> evidence FOR "Effective"; negative weight -> evidence FOR "Ineffective".
# The magnitude is LIME's locally-fit linear weight, not a probability, so only relative
# ranking/direction (not the exact number) should be interpreted clinically.
print("\nLIME explanation for patient 0 (top feature weights and directions):")
for feature_condition, weight in lime_explanation.as_list():
    direction = "-> pushes toward Effective" if weight > 0 else "-> pushes toward Ineffective"
    print(f"  {feature_condition:<45} weight={weight:+.4f}  {direction}")

lime_figure = lime_explanation.as_pyplot_figure()
lime_figure.savefig(OUTPUT_DIR / "lime_explanation_patient0.png", dpi=150, bbox_inches="tight")
plt.close(lime_figure)

print("\nSaved: outputs/shap_summary.png, outputs/shap_force_patient0.png, outputs/lime_explanation_patient0.png")