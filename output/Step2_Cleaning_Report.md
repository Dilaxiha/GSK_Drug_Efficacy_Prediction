# Step 2: Data Cleaning and Preprocessing Report

## Dataset Metrics

- **Input:** `C:\Users\Owner\BCP_3\clinical_data_raw.csv`
- **Output:** `C:\Users\Owner\BCP_3\dataset\clinical_data_cleaned.csv`
- **Before:** 1,155,000 rows x 41 columns
- **After:** 1,015,405 rows x 32 columns
- **Target-missing rows dropped:** 70,592
- **Duplicate rows removed:** 69,003
- **Dropped columns:** patient_id, age, gender, drug_name, weight_lbs, extra_col_1, extra_col_2, unnamed_0, notes

## Processing Applied

1. Stripped column-header whitespace and consolidated duplicate variants case-insensitively using `combine_first`.
2. Standardized headers to lowercase and removed redundant, empty, and free-text columns.
3. Converted placeholders to missing values, normalized gender and ethnicity, and mapped `treatment_outcome` to binary values.
4. Coerced numeric fields to `float64`.
5. Calculated 1.5 x IQR bounds, clipped only the requested continuous fields, then imputed numeric medians and categorical modes.
6. Removed exact duplicate rows after consolidation.

## Outlier Clipping

| Column | Lower bound | Upper bound | Values clipped |
| --- | --- | --- | --- |
| age | 4.5 | 104.5 | 26,112 |
| weight_kg | 28.45 | 127.25 | 21,694 |
| bmi | 5.85 | 48.65 | 71,690 |
| hemoglobin | 7.5 | 19.5 | 15,802 |
| wbc_count | 660 | 14332 | 15,881 |
| alt_enzyme | -14.6 | 67.8 | 50,777 |
| ast_enzyme | -7.375 | 54.425 | 42,299 |
| creatinine | -0.075 | 2.365 | 34,161 |
| egfr | 6.85 | 143.25 | 15,764 |
| hba1c | 1.7 | 11.3 | 17,475 |
| dosage | -700 | 1220 | 0 |
| duration_days | -127.5 | 268.5 | 104,154 |

## Imputation

| Column | Method | Value | Missing before |
| --- | --- | --- | --- |
| age | median | 54.0 | 62,653 |
| weight_kg | median | 77.9 | 54,377 |
| height_cm | median | 169.7 | 161,640 |
| bmi | median | 26.9 | 80,754 |
| systolic_bp | median | 130.0 | 97,472 |
| diastolic_bp | median | 79.0 | 55,440 |
| heart_rate | median | 75.0 | 122,538 |
| temperature_f | median | 98.6 | 126,230 |
| hemoglobin | median | 13.5 | 102,768 |
| wbc_count | median | 7499.0 | 112,139 |
| alt_enzyme | median | 24.5 | 55,174 |
| ast_enzyme | median | 22.2 | 58,769 |
| creatinine | median | 1.11 | 134,896 |
| egfr | median | 75.0 | 119,506 |
| hba1c | median | 6.5 | 58,639 |
| total_cholesterol | median | 200.0 | 135,098 |
| dosage | median | 100.0 | 161,928 |
| duration_days | median | 60.0 | 157,795 |
| concurrent_drugs | median | 3.0 | 140,034 |
| readmission_30d | median | 0.0 | 78,702 |
| ethnicity | mode | Hispanic | 279,742 |
| blood_pressure | mode | 126/83 | 115,962 |
| drug_name | mode | Warfarine | 140,529 |
| route | mode | ORAL | 311,548 |
| diagnosis | mode | T2DM | 204,631 |
| smoking_status | mode | No | 319,800 |
| alcohol_use | mode | daily | 416,359 |
| admission_date | mode | 2020 | 314,805 |
| adverse_event | mode | 0 | 85,858 |

## Validation

- Target values restricted to `0` and `1`: **PASS**
- Gender values restricted to `M`, `F`, `Other`: **PASS**
- Duplicate rows remaining: **0**
- Remaining missing cells: **0**
