import os
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from fraud_utils import FEATURES, TARGET, train_and_save_model, load_model, evaluate_model

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")
st.title("🛡️ FraudShield AI")
st.caption("Machine-learning prototype for identifying potentially fraudulent card transactions.")

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.joblib"

with st.sidebar:
    st.header("Project setup")
    st.write("1. Download the Kaggle credit-card fraud dataset.")
    st.write("2. Save the CSV as `data/creditcard.csv`.")
    st.write("3. Click **Train / refresh model**.")
    st.divider()
    if st.button("Train / refresh model", type="primary"):
        if not os.path.exists(DATA_PATH):
            st.error("Dataset not found. Add data/creditcard.csv first.")
        else:
            with st.spinner("Training model..."):
                metrics = train_and_save_model(DATA_PATH, MODEL_PATH)
            st.session_state["metrics"] = metrics
            st.success("Model trained and saved.")

if not os.path.exists(DATA_PATH):
    st.warning("Dataset not found. Download `creditcard.csv` from the Kaggle dataset linked in README and place it inside the `data` folder.")
    st.stop()

df = pd.read_csv(DATA_PATH)
required = set(FEATURES + [TARGET])
if not required.issubset(df.columns):
    st.error("Dataset columns do not match the expected Kaggle credit-card fraud dataset.")
    st.stop()

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if model is None:
    st.info("No trained model found. Use **Train / refresh model** in the sidebar.")
    st.stop()

total = len(df)
fraud_count = int(df[TARGET].sum())
legit_count = total - fraud_count
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total transactions", f"{total:,}")
c2.metric("Fraud cases in dataset", f"{fraud_count:,}")
c3.metric("Legitimate cases", f"{legit_count:,}")
c4.metric("Fraud rate", f"{(fraud_count / total * 100):.3f}%")

st.subheader("Model evaluation")
metrics = st.session_state.get("metrics")
if metrics is None:
    metrics = evaluate_model(DATA_PATH, MODEL_PATH)
if metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision (fraud)", f"{metrics['precision']:.3f}")
    m2.metric("Recall (fraud)", f"{metrics['recall']:.3f}")
    m3.metric("F1-score (fraud)", f"{metrics['f1']:.3f}")
    st.caption("Metrics are calculated on a held-out test split. Results depend on the dataset and model settings.")

st.subheader("Transaction data")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Upload transactions for prediction")
st.write("Upload a CSV with the same feature columns as the Kaggle dataset. The `Class` column is optional.")
uploaded = st.file_uploader("Choose a transaction CSV", type=["csv"])
if uploaded:
    incoming = pd.read_csv(uploaded)
    missing = [col for col in FEATURES if col not in incoming.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        X_new = incoming[FEATURES].copy()
        probabilities = model.predict_proba(X_new)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results = incoming.copy()
        results["fraud_probability"] = probabilities
        results["risk_score"] = (probabilities * 100).round(1)
        results["prediction"] = ["Potential fraud" if p else "Likely legitimate" for p in predictions]
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False).encode("utf-8"),
            file_name="fraudshield_predictions.csv",
            mime="text/csv"
        )

st.subheader("Important limitations")
st.warning(
    "This is an educational prototype, not a production payment-security system. "
    "It does not block payments or connect to banks. Review alerts manually, "
    "protect transaction data, and validate the model before any real-world use."
)
