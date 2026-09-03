# GSK-Medicine-Outcome-Prediction

**UC-01: Clinical Documentation Case Study | Medicine Outcome Prediction System**  
*GlaxoSmithKline (GSK) — Healthcare & Pharmaceuticals*

> An enterprise-grade machine learning pipeline engineered to predict clinical treatment efficacy across 1M+ patient records with strict data leakage protection. Features multi-model benchmarking (XGBoost, Gradient Boosting, Deep Learning), dynamic threshold optimization, and automated overfitting diagnostics for imbalanced clinical data.

---

## Executive Overview

The **Medicine Outcome Prediction System (UC-01)** is a high-performance clinical AI pipeline designed to evaluate treatment success (`treatment_outcome`) across large-scale electronic health records. Grounded in statistical rigor, the pipeline handles severe target class imbalance (~75% Ineffective / ~25% Effective) by leveraging dynamic threshold tuning, balance-aware loss functions, and multi-metric overfitting analysis.

### Core Architecture Highlights
* **Leakage-Safe Data Flow:** Enforces a strict 60/20/20 Stratified Split (Train/Validation/Test). Preprocessing and dynamic threshold optimizations are computed exclusively on non-test partitions.
* **Multi-Model Benchmarking:** Evaluates Decision Trees, Random Forests, XGBoost, Gradient Boosting, Keras MLPs, and LSTM networks under identical conditions.
* **Multi-Metric Overfitting Diagnostics:** Automatically tracks train-vs-test performance gaps across PR-AUC, F1-Score, and ROC-AUC to prevent model memorization.
* **Automated Executive Reporting:** Generates individual diagnostic `.txt` files and a multi-page PDF benchmark report summarizing ROC/PR curves and model rankings.

---

## Performance Benchmark Summary

All models were evaluated on an unseen test set of **203,081 clinical records** with dynamic decision threshold tuning applied to maximize minority class F1-Score and Recall.

| Rank | Model | PR-AUC | F1-Score | Recall | ROC-AUC | Accuracy | Precision | Threshold | Overfitting Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Gradient Boosting** | **0.3473** | **0.4189** | **0.7345** | **0.6576** | **0.5824** | **0.2931** | **0.4582** | **Well-Balanced** |
| **2** | **XGBoost** | 0.3466 | 0.4189 | 0.7102 | 0.6571 | 0.5981 | 0.2965 | 0.4704 | **Well-Balanced** |
| **3** | **Random Forest** | 0.3436 | 0.4171 | 0.7436 | 0.6555 | 0.5694 | 0.2891 | 0.4546 | **Moderate Overfit** |
| **4** | **Decision Tree** | 0.3351 | 0.4138 | 0.7290 | 0.6481 | 0.5741 | 0.2882 | 0.4549 | **Well-Balanced** |
| **5** | **Keras MLP** | 0.3411 | 0.4127 | 0.7041 | 0.6483 | 0.5525 | 0.2919 | 0.3646 | **Well-Balanced** |
| **6** | **LSTM Network** | 0.2231 | 0.3651 | 0.9999 | 0.4989 | 0.2234 | 0.2234 | 0.2693 | **Trivial Collapse** |

*Note: PR-AUC and F1-Score serve as the primary ranking metrics due to class imbalance.*

---

## Repository Structure

```text
GSK-Medicine-Outcome-Prediction/
├── dataset/
│   └── feature_engineered_clinical_data.csv    # Feature-engineered clinical records
├── metrics_output/                             # Model metric JSONs & individual TXT reports
├── reports/
│   └── Clinical_Model_Performance_Report.pdf   # Executive benchmark report (PDF)
├── utils.py                                    # Data splitters, threshold tuners & evaluation core
├── 01_fit_decision_tree.py                     # Decision Tree implementation
├── 02_fit_random_forest.py                    # Random Forest implementation
├── 03_fit_xgboost.py                          # XGBoost implementation
├── 04_fit_gradient_boosting.py                # Gradient Boosting implementation
├── 05_evaluate_and_report.py                   # Master benchmark & PDF report generator
├── 06_fit_keras_mlp.py                         # Keras Multi-Layer Perceptron
├── 07_fit_lstm.py                              # Sequential LSTM model
├── requirements.txt                            # Python dependency specifications
└── README.md                                   # Project documentation