"""
Customer Churn Prediction — full pipeline
EDA -> preprocessing -> model training -> evaluation -> feature importance
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
OUT = "/home/claude/churn_project/outputs"

# ---------- 1. Load data ----------
df = pd.read_csv("/home/claude/churn_project/data/synthetic_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)

print(f"Rows: {len(df)}, Churn rate: {df['Churn_flag'].mean():.1%}")

# ---------- 2. EDA plots ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Churn by Contract Type")

sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Churn by Tenure")

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 0])
axes[1, 0].set_title("Monthly Charges vs Churn")

churn_by_internet = df.groupby("InternetService")["Churn_flag"].mean().sort_values()
churn_by_internet.plot(kind="barh", ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Churn Rate by Internet Service")
axes[1, 1].set_xlabel("Churn Rate")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=150)
plt.close()
print("Saved eda_overview.png")

# ---------- 3. Preprocessing ----------
target = "Churn_flag"
drop_cols = ["customerID", "Churn", "Churn_flag"]
X = df.drop(columns=drop_cols)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 4. Train models ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
}

results = []
roc_data = {}
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", model)])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, proba),
    })

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr)

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\nModel comparison:")
print(results_df.to_string(index=False))

# ---------- 5. ROC curve plot ----------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = results_df.loc[results_df.Model == name, "ROC-AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()
print("Saved roc_curves.png")

# ---------- 6. Best model: confusion matrix + feature importance ----------
best_name = results_df.iloc[0]["Model"]
best_pipe = fitted_pipelines[best_name]
preds = best_pipe.predict(X_test)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.title(f"Confusion Matrix — {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")

# Feature importance (tree models only)
if hasattr(best_pipe.named_steps["clf"], "feature_importances_"):
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    importances = best_pipe.named_steps["clf"].feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    fi.sort_values().plot(kind="barh", color="darkorange")
    plt.title(f"Top 15 Feature Importances — {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
    plt.close()
    fi.to_csv(f"{OUT}/feature_importance.csv")
    print("Saved feature_importance.png/.csv")
    print("\nTop 10 features driving churn:")
    print(fi.head(10).to_string())

with open(f"{OUT}/classification_report.txt", "w") as f:
    f.write(f"Best model: {best_name}\n\n")
    f.write(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

print(f"\nBest model: {best_name}")
print("All outputs saved to", OUT)
