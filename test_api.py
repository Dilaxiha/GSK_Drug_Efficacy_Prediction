"""
test_api.py - Client test suite for the Medicine Outcome Prediction API.
Run: python app.py   (in one terminal)
     python test_api.py  (in another)
"""
import requests

BASE_URL = "http://localhost:5000"
TIMEOUT = 10

VALID_PATIENT = {
    "age": 55, "gender_encoded": 1, "bmi": 28.5, "dosage_mg": 250,
    "hemoglobin": 13.5, "creatinine": 1.2, "egfr": 65, "hba1c": 7.1,
    "concurrent_drugs": 3, "liver_risk": 0, "polypharmacy": 0,
}


def test_health() -> bool:
    """GET /health -> 200, model_loaded == True."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        body = response.json()
        ok = response.status_code == 200 and body.get("model_loaded") is True
        print(f"[{'PASS' if ok else 'FAIL'}] Health check -> {response.status_code} {body}")
        return ok
    except requests.RequestException as exc:
        print(f"[FAIL] Health check -> request error: {exc}")
        return False


def test_valid_prediction() -> bool:
    """POST /predict with a complete, valid payload -> 200 with all expected keys."""
    try:
        response = requests.post(f"{BASE_URL}/predict", json=VALID_PATIENT, timeout=TIMEOUT)
        body = response.json()
        required_keys = {"prediction", "probability", "confidence", "model"}
        ok = response.status_code == 200 and required_keys.issubset(body)
        print(f"[{'PASS' if ok else 'FAIL'}] Valid prediction -> {response.status_code} {body}")
        return ok
    except requests.RequestException as exc:
        print(f"[FAIL] Valid prediction -> request error: {exc}")
        return False


def test_missing_field() -> bool:
    """POST /predict with a required feature omitted -> 400."""
    payload = {key: value for key, value in VALID_PATIENT.items() if key != "bmi"}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        ok = response.status_code == 400
        print(f"[{'PASS' if ok else 'FAIL'}] Missing field -> {response.status_code} {response.json()}")
        return ok
    except requests.RequestException as exc:
        print(f"[FAIL] Missing field -> request error: {exc}")
        return False


def test_non_numeric_field() -> bool:
    """POST /predict with a string in place of a numeric feature -> 400."""
    payload = {**VALID_PATIENT, "age": "not-a-number"}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        ok = response.status_code == 400
        print(f"[{'PASS' if ok else 'FAIL'}] Non-numeric field -> {response.status_code} {response.json()}")
        return ok
    except requests.RequestException as exc:
        print(f"[FAIL] Non-numeric field -> request error: {exc}")
        return False


def main() -> None:
    tests = {
        "Health check": test_health,
        "Valid prediction": test_valid_prediction,
        "Missing field -> 400": test_missing_field,
        "Non-numeric field -> 400": test_non_numeric_field,
    }
    results = {name: test_fn() for name, test_fn in tests.items()}

    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    for name, passed in results.items():
        print(f"{'PASS' if passed else 'FAIL':<6} {name}")
    passed_count = sum(results.values())
    print(f"\n{passed_count}/{len(results)} tests passed")


if __name__ == "__main__":
    main()

