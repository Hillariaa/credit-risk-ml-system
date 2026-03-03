```markdown

#  Credit Risk ML System  
### Production-Oriented Machine Learning Service for Credit Default Prediction

---

##  Overview

This project implements a **production-style credit risk modeling system**, built beyond notebook experimentation.

It includes:

- ✅ Reproducible ML training pipeline  
- ✅ Threshold optimization under recall constraints  
- ✅ Model versioning and metadata tracking  
- ✅ Explainability via feature contribution analysis  
- ✅ Prediction logging  
- ✅ Statistical drift detection  
- ✅ Dockerized deployment with pinned dependencies  

The goal is to simulate how a real-world financial risk system is engineered, monitored, and maintained.

---

##  Problem Statement

Given applicant financial and behavioral features, predict the probability of default and make an approval/rejection decision based on an optimized threshold.

Unlike demo notebooks, this system focuses on:

- Reliability  
- Observability  
- Governance  
- Deployment discipline  

---

##  System Architecture

Training Layer
├── Data Split
├── Pipeline (Imputer → Scaler → Logistic Regression)
├── Threshold Optimization
└── Metadata + Baseline Statistics

Model Artifacts
├── credit_model.pkl
├── threshold.json
├── baseline_stats.json
└── metadata.json

Serving Layer (FastAPI)
├── /predict
├── /health
├── Explainability
└── Prediction Logging

Monitoring Layer
└── Drift Detection (Z-score)

---

##  Training Pipeline

The training system:

- Uses `StandardScaler` and `LogisticRegression` inside a pipeline  
- Performs stratified train-test split  
- Computes ROC-AUC  
- Searches for a decision threshold under the constraint:

> **Recall ≥ 80%**

It saves:

- Model artifact  
- Selected threshold  
- Baseline feature statistics  
- Training metadata (version, timestamp, metrics)

Example output:

ROC-AUC: 0.9997
Chosen Threshold: 0.84
Model Version: c6b126c8


---

##  Explainability

The API returns:

- Default probability  
- Decision (approve/reject)  
- Top risk drivers  
- Risk-reducing factors  
- Model version  

Explainability is computed via scaled logistic regression contributions, enabling inspection of which features push risk upward or downward.

Example response:

json:

{
  "model_version": "c6b126c8",
  "default_probability": 0.78,
  "decision": 0,
  "top_risk_drivers": [...],
  "risk_reducing_factors": [...]
}

---

#  Monitoring & Drift Detection

Each prediction is logged.

A monitoring script compares live traffic distribution to training baseline using Z-score analysis:

Z = |live_mean - baseline_mean| / baseline_std

if

Z > 2

⚠ Drift Detected

This simulates real-world data shift monitoring in production ML systems.

---

#  Model Versioning

Each training run generates:

Unique model version ID

Timestamp

ROC-AUC

Threshold used

Feature schema

This ensures:

Traceability

Auditability

Safe rollback capability

---

 #  Deployment (Dockerized)

The system is containerized with:

Pinned dependencies (scikit-learn==1.6.0)

Reproducible environment

FastAPI served on port 8000

Build:

docker build -t credit-risk-ml-system .

Run:

docker run -p 8000:8000 credit-risk-ml-system

Access:

http://localhost:8000/docs

---

#  Project Strucure

credit-risk-ml-system/
├── data/
├── models/              # generated artifacts (gitignored)
├── src/
│   ├── app.py
│   ├── train.py
│   ├── monitor.py
│   ├── model.py
│   └── generate_data.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore

---

#  Engineering Decisions & Trade-offs

Why Logistic Regression?
Interpretable, stable, and appropriate for structured financial risk modeling.

Why Threshold Optimization?
Accuracy alone is misleading in imbalanced datasets.
Recall constraint ensures risk capture while maximizing precision.

Why Versioning?
Model artifacts must be traceable in regulated systems.

Why Drift Monitoring?
Production data distributions change.
Unmonitored models silently degrade.

Why Dependency Pinning?
Model serialization is version-sensitive.
Reproducibility prevents runtime failure.

---

#  What This Project Demonstrates

Production-aware ML system design

Governance-conscious modeling

Deployment discipline

Monitoring mindset

Engineering trade-off thinking

This project is intentionally built beyond notebook experimentation to reflect real-world system constraints.

---

#  Planned Improvements

Automated retraining trigger on drift

CI/CD pipeline

Cloud deployment (AWS/GCP)

Structured JSON logging

Metrics dashboard

---

# Author

**Hilary Azimoh**

AI Engineering Portfolio Project
