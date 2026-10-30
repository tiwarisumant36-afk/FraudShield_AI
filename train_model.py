from fraud_utils import train_and_save_model

if __name__ == "__main__":
    metrics = train_and_save_model("data/creditcard.csv", "models/fraud_model.joblib")
    print("Model training completed.")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
