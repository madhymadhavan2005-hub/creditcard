import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

print("Generating Amount/Time-driven training dataset...")
np.random.seed(42)
n_samples = 3000

# Generate distributions for Time and Amount
amounts = np.random.exponential(scale=200.0, size=n_samples) + 10.0
times = np.random.uniform(0, 86400, size=n_samples)

df = pd.DataFrame({"Time": times, "Amount": amounts})

# Automatically define clear fraud logic based on Amount thresholds
df["Class"] = 0
fraud_mask = df["Amount"] > 400.0  # Any transaction over $400 is flagged as fraud
df.loc[fraud_mask, "Class"] = 1

X = df[["Time", "Amount"]]
feature_order = list(X.columns)

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_order)

# Removed stratify to prevent class count errors
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, df["Class"], test_size=0.2, random_state=42
)

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)

# Save artifacts
joblib.dump(model, "xgb_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_order, "feature_order.pkl")
df.head(10).to_csv("creditcard.csv", index=False)

print(
    "✅ Success! xgb_fraud_model.pkl, scaler.pkl, feature_order.pkl, and"
    " creditcard.csv generated."
)