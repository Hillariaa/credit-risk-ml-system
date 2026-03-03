import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

income = np.random.normal(60000, 20000, n).clip(20000, 200000)
age = np.random.randint(21, 70, n)
debt_ratio = np.random.uniform(0.1, 0.9, n)
employment_years = np.random.randint(0, 40, n)
credit_history_length = np.random.randint(1, 30, n)
past_defaults = np.random.poisson(0.3, n)

# Create risk score
risk_score = (
    0.00002 * (100000 - income) +
    1.5 * debt_ratio +
    -0.02 * employment_years +
    0.5 * past_defaults
)

probability = 1 / (1 + np.exp(-risk_score))
default = (probability > 0.5).astype(int)

df = pd.DataFrame({
    "income": income,
    "age": age,
    "debt_ratio": debt_ratio,
    "employment_years": employment_years,
    "credit_history_length": credit_history_length,
    "past_defaults": past_defaults,
    "default": default
})

df.to_csv("data/credit_data.csv", index=False)

print("Synthetic dataset created.")