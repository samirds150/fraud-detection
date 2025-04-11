import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_valid():
    payload = {
        "amount": 450.0,
        "account_age_days": 120,
        "time_of_day": 15,
        "ip_risk_score": 0.8,
        "mouse_movement_score": 0.67,
        "is_vpn": 1,
        "is_international": 0,
        "avg_transaction_amount": 320.0,
        "day_of_week": "Monday"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "fraud_prediction" in response.json()
    assert response.json()["fraud_prediction"] in [0, 1]

def test_predict_invalid_schema():
    payload = {
        # Missing required field: 'amount'
        "account_age_days": 50,
        "time_of_day": 12,
        "ip_risk_score": 0.3,
        "mouse_movement_score": 0.75,
        "is_vpn": 0,
        "is_international": 1,
        "avg_transaction_amount": 150.0,
        "day_of_week": "Wednesday"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
