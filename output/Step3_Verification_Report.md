# Step 3: Post-Cleaning Verification and Exploratory Data Analysis

## Dataset and Integrity

- **Input:** `C:\Users\Owner\BCP_3\dataset\clinical_data_cleaned.csv`
- **Dimensions:** 1,015,405 rows x 32 columns
- **Missing-cell assertion:** **PASS** (0 cells)
- **Binary-target assertion:** **PASS**
- **NZV features (>99% dominant value):** 0
- **Quasi-complete separation categories:** 0

### Target Class Balance

| label | count | rate |
| --- | --- | --- |
| 0.0 | 788,599 | 77.66% |
| 1.0 | 226,806 | 22.34% |

## Continuous Summary Statistics

Post-cleaning summary for requested physiological fields:

| feature | mean | std | min | 25% | median | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| age | 54.3987 | 18.536 | 4.5 | 43 | 54 | 66 | 104.5 |
| weight_kg | 77.7755 | 18.2798 | 28.45 | 66.3 | 77.9 | 89.5 | 127.25 |
| bmi | 27.4229 | 8.63533 | 5.85 | 22.3 | 26.9 | 32 | 48.65 |
| hemoglobin | 13.4881 | 2.14786 | 7.5 | 12.2 | 13.5 | 14.8 | 19.5 |
| wbc_count | 7496.58 | 2431.74 | 660 | 6013 | 7499 | 8977 | 14332 |
| alt_enzyme | 28.2352 | 15.8052 | -5 | 16.7 | 24.5 | 36 | 67.8 |
| ast_enzyme | 24.5588 | 11.714 | -5 | 16.1 | 22.2 | 30.5 | 54.425 |
| creatinine | 1.17604 | 0.442385 | -0.075 | 0.88 | 1.11 | 1.39 | 2.365 |
| egfr | 75.0036 | 24.1889 | 6.85 | 60.4 | 75 | 89.7 | 143.25 |
| hba1c | 6.50147 | 1.79258 | 1.7 | 5.4 | 6.5 | 7.7 | 11.3 |
| dosage | 239.373 | 304.715 | -1 | 25 | 100 | 250 | 1000 |
| duration_days | 83.7337 | 78.0896 | -1 | 21 | 60 | 120 | 268.5 |

## Basic Profiling

### Numeric Feature Profile

| feature | dtype | unique_values | missing | mean | std |
| --- | --- | --- | --- | --- | --- |
| patient_id | float64 | 985876 | 0 | 525138 | 303057 |
| age | float64 | 7585 | 0 | 54.3987 | 18.536 |
| weight_kg | float64 | 8617 | 0 | 77.7755 | 18.2798 |
| height_cm | float64 | 9301 | 0 | 167.158 | 24.1161 |
| bmi | float64 | 7535 | 0 | 27.4229 | 8.63533 |
| systolic_bp | float64 | 225 | 0 | 129.547 | 23.8255 |
| diastolic_bp | float64 | 140 | 0 | 79.4508 | 14.6073 |
| heart_rate | float64 | 140 | 0 | 75.6513 | 27.038 |
| temperature_f | float64 | 80 | 0 | 98.3731 | 5.01948 |
| hemoglobin | float64 | 7435 | 0 | 13.4881 | 2.14786 |
| wbc_count | float64 | 13661 | 0 | 7496.58 | 2431.74 |
| alt_enzyme | float64 | 665 | 0 | 28.2352 | 15.8052 |
| ast_enzyme | float64 | 527 | 0 | 24.5588 | 11.714 |
| creatinine | float64 | 224 | 0 | 1.17604 | 0.442385 |
| egfr | float64 | 1366 | 0 | 75.0036 | 24.1889 |
| hba1c | float64 | 98 | 0 | 6.50147 | 1.79258 |
| total_cholesterol | float64 | 396 | 0 | 203.493 | 81.932 |
| dosage | float64 | 13 | 0 | 239.373 | 304.715 |
| duration_days | float64 | 11 | 0 | 83.7337 | 78.0896 |
| concurrent_drugs | float64 | 19 | 0 | 3.28057 | 5.2197 |
| treatment_outcome | float64 | 2 | 0 | 0.223365 | 0.416501 |
| readmission_30d | float64 | 2 | 0 | 0.13922 | 0.346176 |

### Categorical Feature Profile

| feature | unique_values | top_category | top_count | top_rate |
| --- | --- | --- | --- | --- |
| drug_name | 71 | Warfarine | 144499 | 0.142307 |
| route | 23 | ORAL | 323512 | 0.318604 |
| diagnosis | 33 | T2DM | 216956 | 0.213664 |
| gender | 3 | Other | 583356 | 0.574506 |
| ethnicity | 17 | Hispanic | 346113 | 0.340862 |
| smoking_status | 22 | No | 332250 | 0.327209 |
| alcohol_use | 13 | daily | 438516 | 0.431863 |

### Post-Cleaning IQR Outlier Review

| feature | lower_bound | upper_bound | outlier_count | outlier_rate |
| --- | --- | --- | --- | --- |
| age | 8.5 | 100.5 | 29021 | 0.0285807 |
| weight_kg | 31.5 | 124.3 | 23686 | 0.0233267 |
| bmi | 7.75 | 46.55 | 71597 | 0.0705108 |
| hemoglobin | 8.3 | 18.7 | 24628 | 0.0242544 |
| wbc_count | 1567 | 13423 | 25336 | 0.0249516 |
| alt_enzyme | -12.25 | 64.95 | 54556 | 0.0537283 |
| ast_enzyme | -5.5 | 52.1 | 46875 | 0.0461638 |
| creatinine | 0.115 | 2.155 | 50954 | 0.050181 |
| egfr | 16.45 | 133.65 | 26084 | 0.0256883 |
| hba1c | 1.95 | 11.15 | 19816 | 0.0195154 |
| dosage | -312.5 | 587.5 | 156569 | 0.154194 |
| duration_days | -127.5 | 268.5 | 0 | 0 |

### Target-Stratified Continuous Means

| feature | class_0_mean | class_1_mean | mean_difference |
| --- | --- | --- | --- |
| age | 56.1746 | 48.2238 | -7.9508 |
| weight_kg | 77.776 | 77.7737 | -0.00226274 |
| bmi | 27.4214 | 27.4283 | 0.00690451 |
| hemoglobin | 13.4881 | 13.4881 | 8.06264e-05 |
| wbc_count | 7495.54 | 7500.22 | 4.6873 |
| alt_enzyme | 28.2434 | 28.2069 | -0.0364647 |
| ast_enzyme | 24.5596 | 24.5561 | -0.0035491 |
| creatinine | 1.19649 | 1.10494 | -0.0915557 |
| egfr | 74.9841 | 75.0713 | 0.0872029 |
| hba1c | 6.50171 | 6.50063 | -0.00107934 |
| dosage | 239.209 | 239.94 | 0.730173 |
| duration_days | 83.7396 | 83.7131 | -0.0264159 |

## Top Baseline Predictors

Ranked by absolute point-biserial correlation with `treatment_outcome`:

| feature | correlation | absolute_correlation | p_value | leakage_warning |
| --- | --- | --- | --- | --- |
| age | -0.178653 | 0.178653 | 0 | False |
| creatinine | -0.0861989 | 0.0861989 | 0 | False |
| concurrent_drugs | -0.0510603 | 0.0510603 | 0 | False |
| height_cm | -0.00262254 | 0.00262254 | 0.00822576 | False |
| systolic_bp | 0.00166261 | 0.00166261 | 0.0938625 | False |
| egfr | 0.00150152 | 0.00150152 | 0.13027 | False |
| temperature_f | 0.00144093 | 0.00144093 | 0.146506 | False |
| readmission_30d | 0.00129777 | 0.00129777 | 0.190967 | False |
| heart_rate | -0.00126317 | 0.00126317 | 0.203067 | False |
| dosage | 0.000998039 | 0.000998039 | 0.314562 | False |

Target-leakage review flags absolute correlations greater than 0.90. These require investigation before modeling.

## Multicollinearity: VIF Warnings

Features with VIF > 10 should be reviewed during Step 4 Feature Engineering. Consider removing redundant variables, creating a clinically motivated composite, or applying regularization.

No findings.

## High-Cardinality and Quasi-Separation Warnings

Categorical variables with many levels should use frequency encoding, grouped rare levels, or carefully fitted one-hot encoding. Categories with 0% or 100% positive rates may cause separation in logistic models.

No findings.

### Near-Zero Variance Features

No findings.

## Verification Status

- Data integrity: **PASS**
- Statistical audits completed: **PASS**
- Required and supplementary figures generated in `output/figures/`: **PASS**
- Ready for Step 4 review: **PASS**
