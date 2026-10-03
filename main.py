import joblib
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load cleaned data
df = pd.read_csv("data/cleaned_churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# Prepare numeric and categorical data
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_cols),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# Create the complete machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and predict
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("CUSTOMER CHURN PREDICTION RESULTS")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel training completed!")
# Create folder for the trained model
Path("models").mkdir(exist_ok=True)

# Save the trained model
joblib.dump(model, "models/churn_model.pkl")

print("Model saved successfully!")

# Predict churn for a new customer
print("\nNEW CUSTOMER PREDICTION")
print("=" * 40)

# Example customer details
new_customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 140.0
}])

# Make prediction
prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0][1]

if prediction == 1:
    print("Prediction: Customer may leave")
else:
    print("Prediction: Customer may stay")

print(f"Estimated churn probability: {probability * 100:.2f}%")

# ==========================================
# CREATE AND SAVE GRAPHS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create a folder for graphs
Path("graphs").mkdir(exist_ok=True)

# Load the original dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

sns.set_style("whitegrid")

# Graph 1: Churn Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("graphs/churn_distribution.png")
plt.close()

# Graph 2: Churn by Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graphs/churn_by_contract.png")
plt.close()

# Graph 3: Monthly Charges vs Churn
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.savefig("graphs/monthly_charges_vs_churn.png")
plt.close()

print("All three graphs saved successfully in the graphs folder!")