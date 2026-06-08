"""
Supplier Recommendation Engine - Model Training
SAP BTP AI PoC - ProcureArc / @SAPProcurementAI
Author: Nikhil Bhosale
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# ── Load data ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

df = pd.read_csv(os.path.join(DATA_DIR, "vendor_scorecard.csv"))

# ── Feature engineering ────────────────────────────────────────────────────────
# Weighted composite score (business logic)
# Weights designed by procurement domain expertise:
#   Delivery Reliability  30%
#   Quality (low rejection is good)  25%
#   Price Variance (low variance is good)  20%
#   On Time Delivery  15%
#   Invoice Accuracy  10%

df["QUALITY_SCORE"] = 100 - df["QUALITY_REJECTION_PCT"]
df["PRICE_SCORE"]   = 100 - df["PRICE_VARIANCE_PCT"]

df["COMPOSITE_SCORE"] = (
    df["DELIVERY_RELIABILITY_PCT"] * 0.30 +
    df["QUALITY_SCORE"]            * 0.25 +
    df["PRICE_SCORE"]              * 0.20 +
    df["ON_TIME_DELIVERY_PCT"]     * 0.15 +
    df["INVOICE_ACCURACY_PCT"]     * 0.10
)

# ── Features and target ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "DELIVERY_RELIABILITY_PCT",
    "ON_TIME_DELIVERY_PCT",
    "QUALITY_REJECTION_PCT",
    "PRICE_VARIANCE_PCT",
    "AVG_LEAD_TIME_DAYS",
    "INVOICE_ACCURACY_PCT",
    "COMPLAINT_COUNT",
    "LAST_AUDIT_SCORE",
    "CERTIFIED_ISO",
    "YEARS_AS_VENDOR",
    "RESPONSE_TIME_HRS"
]

X = df[FEATURE_COLS]
y = df["COMPOSITE_SCORE"]

# ── Scale features ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── Train model ────────────────────────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X_scaled, y)

print(f"Model trained on {len(df)} vendors")
print(f"Feature importances:")
for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:40s}: {imp:.4f}")

# ── Save model and scaler ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "../model")
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "supplier_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(FEATURE_COLS, f)

print("\nModel saved to /model/supplier_model.pkl")
print("Scaler saved to /model/scaler.pkl")
