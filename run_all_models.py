"""
run_all_models.py
"""

import os
import subprocess
import sys
import time

MODEL_SCRIPTS = [
    "01_fit_decision_tree.py",
    "02_fit_random_forest.py",
    "03_fit_xgboost.py",
    "04_fit_gradient_boosting.py",
]

REPORT_SCRIPT = "05_evaluate_and_report.py"


def run_script(script_name: str) -> bool:
    print(f"\n==================================================", flush=True)
    print(f"  STARTING: {script_name}", flush=True)
    print(f"==================================================", flush=True)

    start_time = time.time()
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"--> [SUCCESS] {script_name} completed in {elapsed:.2f} seconds.")
        return True
    else:
        print(f"--> [FAILED] {script_name} exited with error code {result.returncode}.")
        return False


def main():
    pipeline_start = time.time()
    completed_models = 0

    for script in MODEL_SCRIPTS:
        if os.path.exists(script):
            success = run_script(script)
            if success:
                completed_models += 1

    if completed_models > 0:
        if os.path.exists(REPORT_SCRIPT):
            run_script(REPORT_SCRIPT)

    total_time = time.time() - pipeline_start
    print(f"\n==================================================")
    print(f" Pipeline Finished! Total Execution Time: {total_time / 60:.2f} minutes")
    print(f"==================================================")


if __name__ == "__main__":
    main()