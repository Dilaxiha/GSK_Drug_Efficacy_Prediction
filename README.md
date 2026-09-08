# GSK-Medicine-Outcome-Prediction

**UC-01: Clinical Documentation Case Study | Medicine Outcome Prediction System**  
*GlaxoSmithKline (GSK) — Healthcare & Pharmaceuticals*

> An enterprise-grade machine learning pipeline engineered to predict clinical treatment efficacy across 1M+ patient records with strict data leakage protection. Features multi-model benchmarking (XGBoost, LightGBM, Gradient Boosting, Deep Learning), model interpretability (SHAP/LIME), a deployable Flask REST API, dynamic threshold optimization, and automated overfitting diagnostics for imbalanced clinical data.

---

## Executive Overview

The **Medicine Outcome Prediction System (UC-01)** is a high-performance clinical AI pipeline designed to evaluate treatment success (`treatment_outcome`) across large-scale electronic health records. Grounded in statistical rigor, the pipeline handles severe target class imbalance (~77.7% Ineffective / ~22.3% Effective) by leveraging dynamic threshold tuning, balance-aware loss functions, and multi-metric overfitting analysis.

### Core Architecture Highlights
* **Leakage-Safe Data Flow:** Enforces a strict 60/20/20 Stratified Split (Train/Validation/Test). Preprocessing and dynamic threshold optimizations are computed exclusively on non-test partitions.
* **Multi-Model Benchmarking:** Evaluates Decision Trees, Random Forests, XGBoost, LightGBM, Gradient Boosting, Keras MLPs, and LSTM networks under identical conditions.
* **Multi-Metric Overfitting Diagnostics:** Automatically tracks train-vs-test performance gaps across PR-AUC, F1-Score, and ROC-AUC to prevent model memorization.
* **Model Interpretability:** SHAP (global summary + per-patient force plots) and LIME (independent local explanation cross-check) for the deployed XGBoost model, with clinician-facing color/direction guidance.
* **REST API Deployment:** A lightweight, memory-safe Flask API (`app.py`) serves real-time `/predict` requests from the tuned XGBoost model, with strict input validation, request-size limits, and a `/health` endpoint.
* **Automated Executive Reporting:** Generates individual diagnostic `.txt`/`.json` files, a compact 2-page executive summary PDF, a reflective model-fitting PDF, and a multi-page PDF benchmark report summarizing ROC/PR curves and model rankings.

---

## Performance Benchmark Summary

All models were evaluated on an unseen test set of **203,081 clinical records** with dynamic decision threshold tuning applied to maximize minority class F1-Score. PR-AUC is the primary ranking metric given the class imbalance.

| Rank | Model | PR-AUC | F1-Score | Recall | ROC-AUC | Accuracy | Precision | Threshold | Overfitting Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **XGBoost** ★ | **0.3465** | 0.4165 | 0.6683 | 0.6545 | 0.5817 | 0.3025 | 0.4841 | **Well-Balanced** |
| **2** | **Gradient Boosting** | 0.3464 | 0.4168 | 0.6609 | 0.6545 | 0.5868 | 0.3043 | 0.4912 | **Well-Balanced** |
| **3** | **Keras MLP** | 0.3454 | 0.4174 | 0.7158 | 0.6540 | 0.5537 | 0.2946 | 0.4660 | **Well-Balanced** |
| **4** | **LSTM Network** | 0.3453 | 0.4162 | 0.6842 | 0.6540 | 0.5712 | 0.2990 | 0.4794 | **Well-Balanced** |
| **5** | **Random Forest** | 0.3429 | 0.4160 | 0.6852 | 0.6526 | 0.5702 | 0.2986 | 0.4732 | **Moderate Overfit** |
| **6** | **Decision Tree** | 0.3320 | 0.4118 | 0.7234 | 0.6473 | 0.5385 | 0.2878 | 0.2159 | **Well-Balanced** |
| **7** | **LightGBM** | 0.3164 | 0.4059 | 0.6955 | 0.6345 | 0.5451 | 0.2865 | 0.2516 | **Well-Balanced** |

★ **Recommended for API deployment: XGBoost** — leads on PR-AUC (the primary metric), maintains well-balanced generalization, and offers fast CPU-only inference (`tree_method='hist'`) suited to an 8 GB RAM target. Gradient Boosting is a statistically-tied runner-up (margin of 0.0002). LightGBM under-trained during this run (early stopping halted after only 4 boosting rounds) and is a candidate for a follow-up tuning pass.

*Note: PR-AUC and F1-Score serve as the primary ranking metrics due to class imbalance. See `reports/Executive_Summary_2Page.pdf` for a compact visual summary and `reports/Clinical_Model_Performance_Report.pdf` for the full 6-model deep-dive.*

---

## Model Interpretability (SHAP & LIME)

`SHAP.py` generates clinician-facing explanations for the deployed XGBoost model:
* **Global SHAP summary plot** (`outputs/shap_summary.png`) — ranks features by overall impact; red/blue dot color shows whether a high or low feature value is driving the push toward "Effective" or "Ineffective".
* **Local SHAP force plot** (`outputs/shap_force_patient0.png`) — explains one patient's individual prediction.
* **LIME cross-check** (`outputs/lime_explanation_patient0.png`) — an independent local explanation for the same patient, printed as ranked feature weights/directions.

## REST API Deployment

* `train_deployment_model.py` — trains and saves the lightweight, 11-raw-feature XGBoost model actually served by the API (`outputs/xgboost_best_model.pkl`); run this first if that file is missing.
* `app.py` — Flask API with `POST /predict` (validated JSON input → prediction/probability/confidence) and `GET /health`; includes pre-flight model-load diagnostics, a request-size cap, and sanitized error responses.
* `test_api.py` — client test suite covering health check, a valid prediction, a missing-field request, and a non-numeric-input request, with a pass/fail summary.

---

## Repository Structure

```text
GSK-Medicine-Outcome-Prediction/
├── dataset/
│   ├── clinical_data_cleaned.csv                 # Cleaned clinical records
│   └── feature_engineered_clinical_data.csv      # Feature-engineered clinical records
├── metrics_output/                               # Model metric JSONs, .pkl/.h5 bundles & TXT reports
├── reports/                                      # Charts + generated PDF reports
│   ├── Clinical_Model_Performance_Report.pdf     # Full 6-model deep-dive report
│   └── Executive_Summary_2Page.pdf               # Compact 2-page, 7-model executive summary
├── outputs/                                      # SHAP/LIME images + deployed API model (.pkl)
├── utils.py                                      # Data splitters, threshold tuners & evaluation core
├── 01_fit_decision_tree.py                       # Decision Tree implementation
├── 02_fit_random_forest.py                       # Random Forest implementation
├── 03_fit_xgboost.py                             # XGBoost implementation (HalvingRandomSearchCV + early stopping)
├── 04_fit_gradient_boosting.py                   # Gradient Boosting implementation
├── 05_evaluate_and_report.py                     # Master benchmark & PDF report generator
├── 06_fit_keras_mlp.py                           # Keras Multi-Layer Perceptron
├── 07_fit_lstm.py                                # Sequential LSTM model
├── 08_fit_lightgbm.py                            # LightGBM implementation (HalvingRandomSearchCV + early stopping)
├── SHAP.py                                       # SHAP + LIME model interpretability
├── train_deployment_model.py                     # Trains the lightweight model served by the API
├── app.py                                        # Flask REST API (/predict, /health)
├── test_api.py                                   # API client test suite
├── create_executive_summary_pdf.py               # Compact 2-page executive summary generator
├── create_model_fitting_reflective_pdf.py        # Reflective model-fitting summary generator
├── clinical_sql_queries.py                       # SQLite exploratory queries over the engineered dataset
├── requirements.txt                               # Python dependency specifications
└── README.md                                     # Project documentation
```