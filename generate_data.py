"""
Generates a realistic, synthetic telecom customer churn dataset.

Note: I don't have live internet access in this environment to download the
real Kaggle "Telco Customer Churn" dataset directly. This script builds a
synthetic dataset with the SAME columns and realistic statistical relationships
(e.g., month-to-month + high monthly charges + low tenure -> higher churn),
so the full pipeline runs and produces meaningful, defensible results.

To use the REAL dataset instead (recommended before you publish this on your
resume/GitHub):
1. Go to https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download WA_Fn-UseC_-Telco-Customer-Churn.csv
3. Drop it into churn_project/data/ and re-run 02_train_model.py pointing
   at that file instead of synthetic_churn.csv (same column names).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 7043  # match real dataset size

customer_id = [f"{7000+i:04d}-{''.join(np.random.choice(list('ABCDEFGHIJKLMNOP'), 5))}" for i in range(n)]
gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])
tenure = np.random.exponential(scale=20, size=n).clip(0, 72).astype(int)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    n, p=[0.34, 0.23, 0.22, 0.21]
)
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])

phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], n, p=[0.42, 0.58])
)

def addon(p_yes=0.30):
    return np.where(internet_service == "No", "No internet service",
                     np.random.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]))

online_security = addon(0.29)
online_backup = addon(0.34)
device_protection = addon(0.34)
tech_support = addon(0.29)
streaming_tv = addon(0.38)
streaming_movies = addon(0.39)

base_charge = np.select(
    [internet_service == "No", internet_service == "DSL", internet_service == "Fiber optic"],
    [20, 55, 85]
)
monthly_charges = (base_charge + np.random.normal(0, 8, n)).clip(18, 120).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 50, n)).clip(0).round(2)

# --- Build churn probability from realistic drivers ---
logit = (
    -2.6
    + 1.4 * (contract == "Month-to-month")
    + 0.5 * (contract == "One year")
    + 0.9 * (internet_service == "Fiber optic")
    + 0.6 * (payment_method == "Electronic check")
    - 0.035 * tenure
    + 0.012 * monthly_charges
    - 0.5 * (tech_support == "Yes")
    - 0.5 * (online_security == "Yes")
    + 0.3 * senior_citizen
    - 0.3 * (partner == "Yes")
    + np.random.normal(0, 0.6, n)  # noise
)
prob = 1 / (1 + np.exp(-logit))
churn = np.where(np.random.random(n) < prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

out_path = "/home/claude/churn_project/data/synthetic_churn.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} rows to {out_path}")
print(df["Churn"].value_counts(normalize=True))
