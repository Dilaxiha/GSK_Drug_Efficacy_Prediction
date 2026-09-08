"""
app.py - Flask API for Medicine Outcome Prediction
Run: python app.py
Test: python test_api.py  (or curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{...}')
"""
import logging
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Reject oversized request bodies outright (the payload is just 11 numeric fields) to avoid a
# trivial memory-exhaustion DoS vector on this memory-constrained (8 GB) deployment target.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "xgboost_best_model.pkl"

# Expected feature order (must match training) -- request payloads are validated against this
# exact list/order before being turned into the model's input array.
FEATURE_COLS = [
    "age", "gender_encoded", "bmi", "dosage_mg",
    "hemoglobin", "creatinine", "egfr", "hba1c",
    "concurrent_drugs", "liver_risk", "polypharmacy",
]

# Pre-flight diagnostics + load once at startup rather than per-request: joblib.load is
# expensive and the model is immutable, so a failed load is logged loudly here and reported via
# /health afterward instead of crashing the whole process.
logger.info("Resolved model path: %s", MODEL_PATH)
if not MODEL_PATH.exists():
    logger.error("Model file does not exist at %s -- /predict will return 503 until it does", MODEL_PATH)
    model = None
else:
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception:
        logger.exception("Model file exists but failed to load from %s", MODEL_PATH)
        model = None


def _extract_features(payload: dict) -> tuple[np.ndarray | None, list[str]]:
    """Validate presence + numeric type of every required field; never trust client input."""
    missing = [name for name in FEATURE_COLS if name not in payload]
    if missing:
        return None, [f"Missing fields: {missing}"]
    try:
        values = [float(payload[name]) for name in FEATURE_COLS]
    except (TypeError, ValueError):
        return None, ["All feature values must be numeric"]
    return np.array([values], dtype=np.float32), []


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded"}), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    features, errors = _extract_features(payload)
    if errors:
        return jsonify({"error": errors[0]}), 400

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception:
        # Don't leak internal exception details (stack traces, paths) to the client.
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

    prediction = int(probability >= 0.5)
    distance_from_midpoint = abs(probability - 0.5)
    confidence = "High" if distance_from_midpoint > 0.3 else "Medium" if distance_from_midpoint > 0.15 else "Low"

    return jsonify({
        "prediction": "Effective" if prediction == 1 else "Ineffective",
        "probability": round(probability, 4),
        "confidence": confidence,
        "model": "XGBoost v1.0",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


if __name__ == "__main__":
    # host=0.0.0.0/debug=False: reachable on the local network for lightweight deployment, but
    # with no built-in auth -- put this behind a reverse proxy/auth layer before exposing it
    # beyond a trusted network. debug=False also avoids exposing the Werkzeug interactive
    # debugger, which allows arbitrary code execution if reachable.
    app.run(host="0.0.0.0", port=5000, debug=False)
