import pandas as pd
import numpy as np
import joblib
import json

import datetime
import uuid

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, recall_score, precision_score)

from model import create_pipeline

# Load the dataset
df = pd.read_csv('data/credit_data.csv')

# Define features and target variable
X = df.drop('default', axis=1)
y = df['default']

# Stratified Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create the pipeline
pipeline = create_pipeline()

# Train the model
pipeline.fit(X_train, y_train)

# Save baseline feature statistics for drift monitoring
baseline_stats = {
    "income_mean": float(X["income"].mean()),
    "income_std": float(X["income"].std()),
    "debt_ratio_mean": float(X["debt_ratio"].mean()),
    "debt_ratio_std": float(X["debt_ratio"].std()),
    "age_mean": float(X["age"].mean()),
    "age_std": float(X["age"].std())
}

with open("models/baseline_stats.json", "w") as f:
    json.dump(baseline_stats, f)

# Evaluate the model
y_prob = pipeline.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_prob)
print(f'ROC-AUC: {roc_auc:.4f}')

# Threshold Selection
thresholds = np.linspace(0, 1, 100)

best_threshold = 0.5
best_precision = 0

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)

    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    if recall >= 0.80 and precision > best_precision:
        best_precision = precision
        best_threshold = t

print(f"Chosen Threshold: {best_threshold:.2f}")

# Generate model version
model_version = str(uuid.uuid4())[:8]

training_metadata = {
    "model_version": model_version,
    "training_timestamp": datetime.datetime.now().isoformat(),
    "roc_auc": float(roc_auc),
    "threshold": float(best_threshold),
    "features": list(X.columns)
}

with open("models/metadata.json", "w") as f:
    json.dump(training_metadata, f, indent=4)

print(f"Model Version: {model_version}")

# Save the trained model
joblib.dump(pipeline, 'models/credit_model.pkl')

with open('models/threshold.json', 'w') as f:
    json.dump({'threshold': float(best_threshold)}, f)

print('Model and configuration saved successfully.')    