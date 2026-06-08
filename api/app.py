"""
Supplier Recommendation Engine - Flask REST API
SAP BTP AI PoC - Deployable on SAP AI Core
Author: Nikhil Bhosale / ProcureArc
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# ── Load artefacts ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "../model")
DATA_DIR   = os.path.join(BASE_DIR, "../data")

with open(os.path.join(MODEL_DIR, "supplier_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

# Load master data
vendors_df    = pd.read_csv(os.path.join(DATA_DIR, "vendor_scorecard.csv"))
lfa1_df       = pd.read_csv(os.path.join(DATA_DIR, "lfa1_vendor_master.csv"))
vendors_df    = vendors_df.merge(lfa1_df[["LIFNR","ORT01"]], on="LIFNR", how="left")

# ── Health check ───────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Supplier Recommendation Engine", "version": "1.0.0"})

# ── Main recommendation endpoint ───────────────────────────────────────────────
@app.route("/v1/recommend", methods=["POST"])
def recommend():
    """
    Input JSON:
    {
        "material_group": "MRO",       -- filter by material group (optional)
        "top_n": 5,                    -- number of recommendations (default 5)
        "min_delivery_reliability": 90 -- optional filter threshold
    }

    Returns ranked vendor list with scores and reasoning.
    """
    try:
        payload = request.get_json(force=True)
        material_group = payload.get("material_group", None)
        top_n          = int(payload.get("top_n", 5))
        min_delivery   = float(payload.get("min_delivery_reliability", 0))

        # Filter vendors
        df = vendors_df.copy()

        if material_group:
            df = df[df["MATERIAL_GROUP"] == material_group.upper()]

        if min_delivery > 0:
            df = df[df["DELIVERY_RELIABILITY_PCT"] >= min_delivery]

        if df.empty:
            return jsonify({
                "status": "no_results",
                "message": f"No vendors found for material group '{material_group}'",
                "recommendations": []
            }), 200

        # Score vendors
        X = df[feature_cols]
        X_scaled = scaler.transform(X)
        scores = model.predict(X_scaled)

        df = df.copy()
        df["AI_SCORE"] = np.round(scores, 2)
        df["RANK"]     = df["AI_SCORE"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("AI_SCORE", ascending=False).head(top_n)

        # Build response
        recommendations = []
        for _, row in df.iterrows():
            recommendations.append({
                "rank":                    int(row["RANK"]),
                "vendor_id":               str(row["LIFNR"]),
                "vendor_name":             row["VENDOR_NAME"],
                "location":                row.get("ORT01", "India"),
                "material_group":          row["MATERIAL_GROUP"],
                "ai_score":                float(row["AI_SCORE"]),
                "key_metrics": {
                    "delivery_reliability_pct": float(row["DELIVERY_RELIABILITY_PCT"]),
                    "on_time_delivery_pct":     float(row["ON_TIME_DELIVERY_PCT"]),
                    "quality_rejection_pct":    float(row["QUALITY_REJECTION_PCT"]),
                    "price_variance_pct":       float(row["PRICE_VARIANCE_PCT"]),
                    "avg_lead_time_days":       int(row["AVG_LEAD_TIME_DAYS"]),
                    "last_audit_score":         int(row["LAST_AUDIT_SCORE"]),
                    "iso_certified":            bool(row["CERTIFIED_ISO"])
                },
                "recommendation_reason":   _build_reason(row)
            })

        return jsonify({
            "status":          "success",
            "material_group":  material_group or "ALL",
            "total_vendors_evaluated": len(vendors_df) if not material_group else len(vendors_df[vendors_df["MATERIAL_GROUP"] == material_group.upper()]),
            "recommendations": recommendations
        }), 200

    except Exception as e:
        app.logger.error(f"Error in /v1/recommend: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Vendor detail endpoint ─────────────────────────────────────────────────────
@app.route("/v1/vendor/<vendor_id>", methods=["GET"])
def vendor_detail(vendor_id):
    df = vendors_df[vendors_df["LIFNR"] == vendor_id]
    if df.empty:
        return jsonify({"status": "not_found"}), 404

    row = df.iloc[0]
    X_scaled = scaler.transform([row[feature_cols]])
    score = model.predict(X_scaled)[0]

    return jsonify({
        "vendor_id":    vendor_id,
        "vendor_name":  row["VENDOR_NAME"],
        "ai_score":     round(float(score), 2),
        "scorecard":    row[feature_cols].to_dict(),
        "reasoning":    _build_reason(row)
    }), 200


# ── Helper: build human-readable recommendation reason ─────────────────────────
def _build_reason(row):
    reasons = []
    if row["DELIVERY_RELIABILITY_PCT"] >= 97:
        reasons.append(f"Exceptional delivery reliability at {row['DELIVERY_RELIABILITY_PCT']}%")
    elif row["DELIVERY_RELIABILITY_PCT"] >= 93:
        reasons.append(f"Strong delivery reliability at {row['DELIVERY_RELIABILITY_PCT']}%")

    if row["QUALITY_REJECTION_PCT"] <= 1.0:
        reasons.append(f"Very low rejection rate of {row['QUALITY_REJECTION_PCT']}%")
    elif row["QUALITY_REJECTION_PCT"] <= 2.0:
        reasons.append(f"Good quality with {row['QUALITY_REJECTION_PCT']}% rejection rate")

    if row["PRICE_VARIANCE_PCT"] <= 2.0:
        reasons.append(f"Consistent pricing with only {row['PRICE_VARIANCE_PCT']}% variance")

    if row["CERTIFIED_ISO"] == 1:
        reasons.append("ISO certified vendor")

    if row["LAST_AUDIT_SCORE"] >= 90:
        reasons.append(f"High audit score of {row['LAST_AUDIT_SCORE']}/100")

    if row["YEARS_AS_VENDOR"] >= 10:
        reasons.append(f"Established vendor with {row['YEARS_AS_VENDOR']} years of supply history")

    return ". ".join(reasons) if reasons else "Meets procurement baseline criteria"

from flask import send_from_directory

@app.route("/")
def serve_ui():
    ui_dir = os.path.join(BASE_DIR, "../ui")
    return send_from_directory(ui_dir, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
