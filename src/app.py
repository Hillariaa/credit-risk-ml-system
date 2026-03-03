import os
import datetime
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI()

# Input Schema
class CreditApplicant(BaseModel):
    income: float
    age: int
    debt_ratio: float
    employment_years: int
    credit_history_length: int
    past_defaults: int

import os

# Load the trained model
MODEL_PATH = "models/credit_model.pkl"
CONFIG_PATH = "models/threshold.json"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Train the model first.")

if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("Config file not found. Train the model first.")

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH) as f:
    config = json.load(f)

coefficients = model.named_steps["classifier"].coef_[0]
intercept = model.named_steps["classifier"].intercept_[0]

feature_names = [
    "income",
    "age",
    "debt_ratio",
    "employment_years",
    "credit_history_length",
    "past_defaults"
]

threshold = config["threshold"]

METADATA_PATH = "models/metadata.json"

if not os.path.exists(METADATA_PATH):
    raise RuntimeError("Metadata file missing.")

with open(METADATA_PATH) as f:
    metadata = json.load(f)

model_version = metadata["model_version"]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

# API Endpoints
@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}

@app.post('/predict')
def predict(applicant: CreditApplicant):

    features = np.array([[
        applicant.income,
        applicant.age,
        applicant.debt_ratio,
        applicant.employment_years,
        applicant.credit_history_length,
        applicant.past_defaults
    ]])

    probability = model.predict_proba(features)[0][1]
    decision = int(probability >= threshold)

    # -------- Logging --------
    log_entry = {
        "timestamp": datetime.datetime.now(),
        "income": applicant.income,
        "age": applicant.age,
        "debt_ratio": applicant.debt_ratio,
        "employment_years": applicant.employment_years,
        "credit_history_length": applicant.credit_history_length,
        "past_defaults": applicant.past_defaults,
        "probability": float(probability),
        "decision": decision
    }

    log_df = pd.DataFrame([log_entry])

    print("Logging prediction...")

    log_df.to_csv(
        "prediction_logs.csv",
        mode="a",
        header=not os.path.exists("prediction_logs.csv"),
        index=False
    )

        # -------- Explainability --------
    scaled_features = model.named_steps["scaler"].transform(features)[0]

    raw_contributions = {}

    for name, coef, value in zip(feature_names, coefficients, scaled_features):
        raw_contributions[name] = float(coef * value)

    # Sort by absolute impact
    sorted_features = sorted(
        raw_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_risk_drivers = []
    risk_reducing_factors = []

    for feature, contribution in sorted_features:
        if contribution > 0:
            top_risk_drivers.append({
                "feature": feature,
                "impact": "increases risk"
            })
        elif contribution < 0:
            risk_reducing_factors.append({
                "feature": feature,
                "impact": "reduces risk"
            })

    # Optional: limit to top 3 each
    top_risk_drivers = top_risk_drivers[:3]
    risk_reducing_factors = risk_reducing_factors[:3]

    return {
        "model_version": model_version,
        "default_probability": float(probability),
        "decision": decision,
        "top_risk_drivers": top_risk_drivers,
        "risk_reducing_factors": risk_reducing_factors
    }