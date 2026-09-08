
import sqlite3
import pandas as pd


# ============================================================
# Load Data into SQLite
# ============================================================

df = pd.read_csv("dataset/feature_engineered_clinical_data.csv")

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")

# Load the DataFrame into SQLite
df.to_sql(
    "patients",
    conn,
    index=False,
    if_exists="replace"
)


# ============================================================
# Helper Function to Run SQL Queries
# ============================================================

def run_query(title, sql):
    print(f"\n{title}")
    print("=" * 60)

    result = pd.read_sql_query(sql, conn)

    print(result.to_string(index=False))

    return result


# ============================================================
# Query 1: Treatment Outcome Distribution
# ============================================================

run_query(
    "Q1: Overall Treatment Outcomes",
    """
    SELECT
        treatment_outcome,
        COUNT(*) AS patient_count,
        ROUND(
            COUNT(*) * 100.0 /
            (SELECT COUNT(*) FROM patients),
            2
        ) AS pct
    FROM patients
    GROUP BY treatment_outcome
    ORDER BY treatment_outcome;
    """
)


# ============================================================
# Query 2: Average Outcome by Age Group
# ============================================================

run_query(
    "Q2: Efficacy by Age Group",
    """
    SELECT
        CASE
            WHEN age < 30 THEN 'Under 30'
            WHEN age BETWEEN 30 AND 50 THEN '30-50'
            WHEN age BETWEEN 51 AND 65 THEN '51-65'
            ELSE 'Over 65'
        END AS age_bucket,

        COUNT(*) AS total_patients,

        SUM(treatment_outcome) AS effective_count,

        ROUND(
            AVG(treatment_outcome) * 100,
            2
        ) AS efficacy_pct

    FROM patients

    GROUP BY age_bucket

    ORDER BY efficacy_pct DESC;
    """
)


# ============================================================
# Query 3: Top Drugs by Efficacy Rate
# ============================================================

run_query(
    "Q3: Drug Efficacy Ranking",
    """
    SELECT
        drug_name,

        COUNT(*) AS total_prescribed,

        SUM(treatment_outcome) AS effective,

        ROUND(
            AVG(treatment_outcome) * 100,
            2
        ) AS efficacy_rate

    FROM patients

    GROUP BY drug_name

    HAVING total_prescribed >= 100

    ORDER BY efficacy_rate DESC

    LIMIT 10;
    """
)


# ============================================================
# Query 4: Adverse Events by Drug and Age
# ============================================================

run_query(
    "Q4: ADR Risk by Drug and Age",
    """
    SELECT
        drug_name,

        CASE
            WHEN age > 65 THEN 'Elderly'
            ELSE 'Non-Elderly'
        END AS age_cat,

        COUNT(*) AS patients,

        SUM(adverse_event) AS adr_count,

        ROUND(
            AVG(adverse_event) * 100,
            2
        ) AS adr_rate

    FROM patients

    GROUP BY
        drug_name,
        age_cat

    ORDER BY adr_rate DESC

    LIMIT 15;
    """
)


# ============================================================
# Query 5: High-Risk Patient Identification
# ============================================================

run_query(
    "Q5: High-Risk Patients (Multiple Risk Factors)",
    """
    SELECT
        patient_id,
        age,
        bmi,
        creatinine,
        concurrent_drugs,
        treatment_outcome,
        adverse_event

    FROM patients

    WHERE age > 65
      AND creatinine > 1.5
      AND concurrent_drugs >= 5

    ORDER BY creatinine DESC

    LIMIT 20;
    """
)


# ============================================================
# Query 6: Outcome vs Dosage Level
# ============================================================

run_query(
    "Q6: Outcome vs Dosage Level",
    """
    SELECT
        CASE
            WHEN dosage < 100 THEN 'Low (<100mg)'
            WHEN dosage BETWEEN 100 AND 500
                THEN 'Medium (100-500mg)'
            ELSE 'High (>500mg)'
        END AS dosage_level,

        COUNT(*) AS patients,

        ROUND(
            AVG(treatment_outcome) * 100,
            2
        ) AS efficacy_pct,

        ROUND(
            AVG(adverse_event) * 100,
            2
        ) AS adr_pct

    FROM patients

    GROUP BY dosage_level;
    """
)


# ============================================================
# Close Database Connection
# ============================================================

conn.close()

print("\n" + "=" * 60)
print("All SQL queries completed successfully.")
print("=" * 60)
