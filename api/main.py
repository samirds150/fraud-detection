from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load trained model
model_path = os.path.abspath("models/fraud_model_rf.joblib")
model = joblib.load(model_path)

# Define FastAPI app
app = FastAPI(title="Fraud Detection API")

# Define input schema
class Transaction(BaseModel):
    amount: float
    account_age_days: int
    time_of_day: int
    ip_risk_score: float               # ✅ not 'ip_risk_scores'
    mouse_movement_score: float
    is_vpn: int
    is_international: int
    avg_transaction_amount: float     # ✅ add missing fields
    day_of_week: str                  # ✅ raw, like 'Monday'


@app.post("/predict")
def predict_fraud(txn: Transaction):
    input_dict = txn.model_dump()
    
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # One-hot encode 'day_of_week' (same as training)
    df = pd.get_dummies(df, columns=['day_of_week'])

    # Align feature columns to match training
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  # fill missing expected columns

    df = df[model_features]  # reorder to match training

    prediction = model.predict(df)[0]
    return {'fraud_prediction': int(prediction)}

