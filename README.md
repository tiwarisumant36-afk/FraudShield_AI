# FraudShield AI

An educational machine-learning prototype that classifies credit-card transactions as potentially fraudulent or likely legitimate.

## Features
- Train a Random Forest classifier on the Kaggle credit-card fraud dataset.
- Evaluate precision, recall, F1-score and ROC-AUC.
- Explore transaction data in a Streamlit dashboard.
- Upload a CSV of transactions and download prediction results with risk scores.

## Requirements
- Python 3.10 or newer (Python 3.11 recommended)
- VS Code or another Python editor

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Dataset
Download the `creditcard.csv` file from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it here:
```text
FraudShield_AI/
  data/
    creditcard.csv
```

The expected columns are `Time`, `V1` through `V28`, `Amount`, and `Class` (`1` = fraud, `0` = legitimate).

## Train the model

```powershell
python train_model.py
```

Or launch the dashboard and click **Train / refresh model** in the sidebar.

## Run the dashboard

```powershell
streamlit run app.py
```

Streamlit will open the application in your browser, usually at http://localhost:8501.

## Project structure

```text
FraudShield_AI/
├── app.py
├── fraud_utils.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv        # download separately
└── models/
    └── fraud_model.joblib    # generated after training
```

## Important
This is a learning project, not a production fraud-prevention service. It does not connect to a payment gateway, bank, or live transaction stream, and it must not be used to approve or reject real payments. The dataset is anonymized and does not contain ordinary customer details. Model performance should be assessed with fraud-class precision/recall and tested for data leakage and changing fraud patterns before any operational use.
