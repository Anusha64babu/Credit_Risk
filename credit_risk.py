import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dataset (you can replace with real CSV later)
data = {
    "loan_amount": [50000, 150000, 100000, 120000],
    "income": [200000, 250000, 300000, 280000],
    "credit_score": [720, 580, 650, 690],
    "employment_length": [5, 2, 7, 6],
    "loan_status": ["Safe", "Default", "Safe", "Safe"]
}
df = pd.DataFrame(data)

# Feature engineering
df["debt_to_income"] = df["loan_amount"] / df["income"]

X = df[["loan_amount","income","credit_score","employment_length","debt_to_income"]]
y = (df["loan_status"] == "Default").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "credit_risk_model.pkl")
print("✅ Credit Risk model saved as credit_risk_model.pkl")
