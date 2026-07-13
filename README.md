# Customer Churn Prediction

Predicts which telecom customers are likely to cancel their subscription (churn),
using classification models trained on customer account, billing, and service data.

## Problem
Customer acquisition costs far more than retention. Identifying at-risk customers
before they cancel lets a business intervene early (discounts, outreach, plan
changes) instead of losing revenue.

## Data
Telecom customer dataset (7,043 customers, 26.1% churn rate) with demographics,
account tenure, contract type, billing method, and subscribed services. Modeled
on the structure of the widely-used [IBM/Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

> Note: `data/synthetic_churn.csv` is a synthetically generated dataset built to
> match the real dataset's schema and realistic churn drivers (see
> `data/generate_data.py`). To reproduce results on the original data, download
> the Kaggle CSV and point `train_model.py` at it — column names match exactly.

## Approach
1. **EDA** — explored churn patterns across contract type, tenure, monthly
   charges, and internet service type.
2. **Preprocessing** — imputed missing billing values, one-hot encoded
   categoricals, standardized numeric features.
3. **Modeling** — trained and compared three classifiers: Logistic Regression,
   Random Forest, and Gradient Boosting.
4. **Evaluation** — compared models on accuracy, precision, recall, F1, and
   ROC-AUC (F1/recall prioritized, since missing a churner is costlier than a
   false alarm).
5. **Interpretation** — extracted feature importances to identify the top
   drivers of churn.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.690 | 0.443 | **0.744** | **0.555** | **0.774** |
| Random Forest | 0.703 | 0.452 | 0.662 | 0.537 | 0.767 |
| Gradient Boosting | **0.776** | **0.623** | 0.351 | 0.449 | 0.765 |

**Selected model: Logistic Regression** — best F1 and ROC-AUC, and highest
recall, which matters most here: catching more true churners (even with some
false positives) is more valuable than a slightly higher accuracy that misses
customers about to leave.

### Top churn drivers (from feature importance)
- Contract type — month-to-month customers churn far more than annual/two-year
- Tenure — newer customers are much more likely to churn
- Fiber optic internet service — higher churn than DSL
- Payment via electronic check — associated with higher churn
- Lack of tech support / online security add-ons

## Business recommendation
Target retention offers (discounted annual contracts, free tech support trial)
at customers in their first 6 months on month-to-month, electronic-check
billing — this segment shows the highest predicted churn risk.

## Repo structure
```
churn_project/
├── data/
│   ├── generate_data.py       # builds the synthetic dataset
│   └── synthetic_churn.csv
├── notebook/
│   └── train_model.py         # full pipeline: EDA -> preprocessing -> training -> eval
├── outputs/
│   ├── eda_overview.png
│   ├── roc_curves.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png / .csv
│   ├── model_comparison.csv
│   └── classification_report.txt
└── README.md
```

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python data/generate_data.py       # generates dataset (or swap in real Kaggle CSV)
python notebook/train_model.py     # runs full pipeline, saves all outputs
```

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn
