# Step 1: Raw Data Problem Audit and Initial Diagnostic EDA

## Dataset Overview

- **Input:** `C:\Users\Owner\BCP_3\clinical_data_raw.csv`
- **Raw dimensions:** 1,155,000 rows x 41 columns
- **Exact duplicate rows:** 70,017
- **Audit scope:** Complete raw file. No cleaning, sampling, imputation, or outlier removal was performed.

## Column Data Types

| Column | Pandas dtype |
| --- | --- |
| Patient_ID | int64 |
| patient_id | float64 |
| Age | float64 |
|  age | float64 |
| Gender | str |
| gender  | str |
| Ethnicity | str |
| Weight_kg | float64 |
| Weight_lbs | float64 |
| Height_cm | float64 |
| BMI | float64 |
| blood_pressure | str |
| Systolic_BP | float64 |
| Diastolic_BP | float64 |
| Heart_Rate | float64 |
| Temperature_F | float64 |
| Hemoglobin | float64 |
| WBC_Count | float64 |
| ALT_Enzyme | float64 |
| AST_Enzyme | float64 |
| Creatinine | float64 |
| eGFR | float64 |
| HbA1c | float64 |
| Total_Cholesterol | float64 |
| Drug_Name | str |
| drug_name | str |
| Dosage | str |
| Duration_Days | str |
| Route | str |
| Concurrent_Drugs | float64 |
| Diagnosis | str |
| Smoking_Status | str |
| Alcohol_Use | str |
| Admission_Date | str |
| Treatment_Outcome | str |
| Adverse_Event | str |
| Readmission_30d | float64 |
| Notes | str |
| Extra_Col_1 | float64 |
| Extra_Col_2 | float64 |
| unnamed_0 | float64 |

## Missing Values

| Column | Missing count | Missing rate |
| --- | --- | --- |
| patient_id | 101,482 | 8.79% |
| Age | 69,936 | 6.06% |
|  age | 1,102,738 | 95.48% |
| Gender | 256,193 | 22.18% |
| gender  | 1,128,109 | 97.67% |
| Ethnicity | 234,923 | 20.34% |
| Weight_kg | 57,941 | 5.02% |
| Weight_lbs | 64,152 | 5.55% |
| Height_cm | 172,218 | 14.91% |
| BMI | 86,040 | 7.45% |
| blood_pressure | 123,626 | 10.70% |
| Systolic_BP | 103,846 | 8.99% |
| Diastolic_BP | 58,915 | 5.10% |
| Heart_Rate | 130,529 | 11.30% |
| Temperature_F | 134,530 | 11.65% |
| Hemoglobin | 109,547 | 9.48% |
| WBC_Count | 119,364 | 10.33% |
| ALT_Enzyme | 58,723 | 5.08% |
| AST_Enzyme | 62,550 | 5.42% |
| Creatinine | 143,619 | 12.43% |
| eGFR | 127,167 | 11.01% |
| HbA1c | 62,478 | 5.41% |
| Total_Cholesterol | 143,994 | 12.47% |
| Drug_Name | 161,339 | 13.97% |
| drug_name | 192,545 | 16.67% |
| Dosage | 159,804 | 13.84% |
| Duration_Days | 162,672 | 14.08% |
| Route | 260,312 | 22.54% |
| Concurrent_Drugs | 149,040 | 12.90% |
| Diagnosis | 160,802 | 13.92% |
| Smoking_Status | 266,387 | 23.06% |
| Alcohol_Use | 278,813 | 24.14% |
| Admission_Date | 335,312 | 29.03% |
| Treatment_Outcome | 70,592 | 6.11% |
| Adverse_Event | 90,047 | 7.80% |
| Readmission_30d | 83,839 | 7.26% |
| Notes | 998,781 | 86.47% |
| Extra_Col_1 | 1,155,000 | 100.00% |
| Extra_Col_2 | 1,155,000 | 100.00% |
| unnamed_0 | 118,680 | 10.28% |

## Target Distribution: `treatment_outcome`

| Raw label | Count | Rate |
| --- | --- | --- |
| 0 | 840,304 | 72.75% |
| 1 | 240,475 | 20.82% |
| <NA> | 70,592 | 6.11% |
| no | 1,230 | 0.11% |
| yes | 1,214 | 0.11% |
| effective | 608 | 0.05% |
| ineffective | 577 | 0.05% |

## Outputs

- Complete line-by-line anomaly inventory: `raw_data_problems.txt`
- Raw distributions for age, weight, BMI, hemoglobin, WBC, liver enzymes, creatinine, eGFR, HbA1c, dosage, and duration: `step1_raw_distributions.png`

The anomaly inventory includes missing values, placeholders, formatting whitespace, duplicate headers, type-conversion failures, invalid categorical codes, clinical-bound violations, statistical IQR outliers, duplicate rows, and target imbalance.
