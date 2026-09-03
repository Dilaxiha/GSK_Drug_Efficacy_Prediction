# Data Exploration and Verification Report

- **Input dataset:** `C:\Users\Owner\BCP_3\dataset\feature_engineered_clinical_data.csv`
- **Dimensions:** 1,015,405 rows x 39 columns
- **Missing cells:** 2,461
- **Target values:** ['0.0', '1.0']

## Target Distribution

treatment_outcome
0.0    788599
1.0    226806

## Mutual Information

                  feature  mi_score
                      age  0.011286
         age_group_Senior  0.008024
         ethnicity_Latina  0.007100
 blood_pressure___Other__  0.006169
         concurrent_drugs  0.006094
 admission_date___Other__  0.005940
admission_date_2019/02/26  0.005861
    blood_pressure_117/84  0.005224
        elderly_high_dose  0.005183
    blood_pressure_125/73  0.005078
                   dosage  0.005010
          adverse_event_0  0.004863
      drug_name_IBUPROFEN  0.004845
             gender_Other  0.004844
               liver_risk  0.004511

## Variance Inflation Factor

          feature      vif  warning
              bmi 2.641027    False
       liver_risk 2.411297    False
        weight_kg 2.179913    False
       alt_enzyme 1.987516    False
        height_cm 1.454255    False
       ast_enzyme 1.434012    False
elderly_high_dose 1.390879    False
              age 1.237464    False
 concurrent_drugs 1.156751    False
     polypharmacy 1.156428    False
           dosage 1.153114    False
      systolic_bp 1.000800    False
        wbc_count 1.000634    False
total_cholesterol 1.000588    False
       hemoglobin 1.000522    False

## Categorical Associations

       feature    chi2_stat       p_value  cramers_v
     age_group 30470.106081  0.000000e+00   0.173228
admission_date 21865.881697  4.261877e-12   0.146745
blood_pressure 21350.030688  1.182594e-11   0.145004
 adverse_event   706.836209 1.153409e-151   0.026384
     drug_name    75.885116  2.945680e-01   0.008645
     diagnosis    24.845475  8.122383e-01   0.004947
         route    22.426301  4.347037e-01   0.004700
smoking_status    21.499852  4.288044e-01   0.004601
   alcohol_use    20.869580  5.232318e-02   0.004534
     ethnicity    20.793400  1.865895e-01   0.004525
  kidney_stage     2.839787  4.169921e-01   0.001672
        gender     0.974417  6.143390e-01   0.000980
  bmi_category     0.658609  8.828937e-01   0.000805

## Generated Figures

- `figures/target_class_imbalance.png`
- `figures/mutual_information_scores.png`
- `figures/correlation_heatmap.png`
