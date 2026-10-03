# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company. It uses customer information and service details to identify possible customer churn.

## Objectives

* Analyze customer data.
* Understand customer churn patterns.
* Train a Machine Learning model.
* Predict whether a customer may leave the company.
* Visualize churn patterns using graphs.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

## Machine Learning Algorithm

Logistic Regression

## Dataset

The project uses the Telco Customer Churn dataset, which contains customer information, service details, contract types, monthly charges, and churn status.

## Project Structure

* `main.py` — runs data processing, model training, and prediction.
* `data/raw/` — contains the original dataset.
* `data/cleaned_churn.csv` — contains cleaned data.
* `models/churn_model.pkl` — saved trained model.
* `graphs/` — contains generated graphs.
* `README.md` — project documentation.

## Model Performance

The model achieved approximately **80.55% accuracy** on the test dataset in the current run. Results may vary if the data or model settings change.

## How to Run

1. Install Python.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run `python main.py`.
4. Check the `graphs` folder for the generated graphs.

## Future Improvements

* Improve prediction performance.
* Build a user-friendly dashboard.
* Provide customer retention suggestions.

## Author

Niveathika J
