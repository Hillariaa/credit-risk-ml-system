import pandas as pd
import json

# Load live prediction logs
df = pd.read_csv("prediction_logs.csv")

# Load baseline stats
with open("models/baseline_stats.json") as f:
    baseline = json.load(f)

print("Total Predictions:", len(df))
print("\n--- Drift Detection ---")

def check_drift(feature):
    live_mean = df[feature].mean()
    baseline_mean = baseline[f"{feature}_mean"]
    baseline_std = baseline[f"{feature}_std"]

    z_score = abs(live_mean - baseline_mean) / baseline_std

    print(f"\nFeature: {feature}")
    print(f"Live Mean: {live_mean}")
    print(f"Baseline Mean: {baseline_mean}")
    print(f"Z-Score Drift: {z_score:.2f}")

    if z_score > 2:
        print("⚠️ Drift Detected")
    else:
        print("No significant drift")

check_drift("income")
check_drift("debt_ratio")
check_drift("age")