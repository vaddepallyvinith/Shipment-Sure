from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import math
from datetime import datetime

app = Flask(__name__)

# Load trained model and feature names (V2)
with open("rf_delivery_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names_v2.txt", "r") as f:
    feature_names = [line.strip() for line in f]

def expand_features(raw_data):
    """
    Expands raw user inputs into the exact 93-feature vector used for training.
    Ensures 100% parity with the training pipeline including bins and risk scores.
    """
    # Initialize payload with 0.0 for all known features
    payload = {f: 0.0 for f in feature_names}
    
    # 1. Direct Mappings
    payload["supplier_lead_time"] = float(raw_data.get("supplier_lead_time") or 0)
    payload["shipping_distance_km"] = float(raw_data.get("shipping_distance_km") or 0)
    payload["order_quantity"] = float(raw_data.get("order_quantity") or 0)
    payload["unit_price"] = float(raw_data.get("unit_price") or 0)
    payload["total_order_value"] = payload["order_quantity"] * payload["unit_price"]
    payload["previous_on_time_rate"] = float(raw_data.get("previous_on_time_rate") or 0.5)
    payload["Promised_Lead_Time"] = float(raw_data.get("Promised_Lead_Time") or 0)

    # 2. Derived Metrics
    payload["supplier_on_time_rate"] = payload["previous_on_time_rate"]
    if payload["previous_on_time_rate"] > 1.0: # Normalize if entered as 0-100
         payload["supplier_on_time_rate"] = payload["previous_on_time_rate"] / 100.0
         
    payload["supplier_delay_rate"] = 1.0 - payload["supplier_on_time_rate"]
    payload["supplier_count"] = 20.0

    # 3. Status Flags
    if payload["supplier_on_time_rate"] >= 0.9: payload["Reliable_Supplier"] = 1.0
    if payload["supplier_on_time_rate"] <= 0.7: payload["Risky_Supplier"] = 1.0
    payload["Experienced_Supplier"] = 1.0

    # --- Heuristic Logistics (Physics-Based) ---
    speed = 500.0 # Road/Default
    mode = raw_data.get("shipment_mode")
    if mode == "Sea": speed = 40.0
    elif mode == "Air": speed = 2000.0
    
    transit_days = payload["shipping_distance_km"] / max(1.0, speed)
    payload["Est_Transit_Days"] = transit_days
    payload["Est_Total_Days"] = payload["supplier_lead_time"] + transit_days
    
    # The 'Gap' - positive means we need MORE time than promised (Delay Likely)
    payload["Time_Gap"] = payload["Est_Total_Days"] - payload["Promised_Lead_Time"]
    
    if payload["Time_Gap"] > 0: payload["Big_Gap_Risk"] = 1.0
    if payload["Time_Gap"] < -2: payload["Safe_Buffer_Excellent"] = 1.0

    # 4. Computed Risk Scores
    payload["Risk_Combo"] = payload["shipping_distance_km"] * payload["supplier_delay_rate"]
    payload["Safe_Combo"] = payload["Promised_Lead_Time"] * payload["supplier_on_time_rate"]
    
    if payload["Risk_Combo"] > 500: payload["Extreme_Risk"] = 1.0
    if payload["Risk_Combo"] > 200: payload["Very_Likely_Delayed"] = 1.0
    if payload["Safe_Combo"] > 10: payload["Almost_Certain_OnTime"] = 1.0
    
    payload["Cost_per_Day"] = payload["total_order_value"] / max(1.0, payload["Promised_Lead_Time"])
    payload["Distance_per_Day"] = payload["shipping_distance_km"] / max(1.0, payload["Promised_Lead_Time"])

    # 5. Mathematical transformations (Full Parity)
    def safe_log(v): return math.log(max(0.0001, v))
    def safe_sqrt(v): return math.sqrt(max(0, v))

    payload["log_Promised_Lead_Time"] = safe_log(payload["Promised_Lead_Time"])
    payload["sqrt_Promised_Lead_Time"] = safe_sqrt(payload["Promised_Lead_Time"])
    payload["log_shipping_distance_km"] = safe_log(payload["shipping_distance_km"])
    payload["sqrt_shipping_distance_km"] = safe_sqrt(payload["shipping_distance_km"])
    payload["log_unit_price"] = safe_log(payload["unit_price"])
    payload["sqrt_unit_price"] = safe_sqrt(payload["unit_price"])
    payload["log_Risk_Combo"] = safe_log(payload["Risk_Combo"])
    payload["sqrt_Risk_Combo"] = safe_sqrt(payload["Risk_Combo"])
    payload["log_supplier_delay_rate"] = safe_log(payload["supplier_delay_rate"])
    payload["sqrt_supplier_delay_rate"] = safe_sqrt(payload["supplier_delay_rate"])

    # 6. Bins and Quantiles (Heuristics)
    # Lead Bins
    if payload["Promised_Lead_Time"] <= 2: payload["Lead_Bin_Fast"] = 1.0
    elif payload["Promised_Lead_Time"] <= 5: payload["Lead_Bin_Normal"] = 1.0
    elif payload["Promised_Lead_Time"] <= 10: payload["Lead_Bin_Slow"] = 1.0
    else: payload["Lead_Bin_VerySlow"] = 1.0

    # Dist Bins
    if payload["shipping_distance_km"] <= 300: payload["Dist_Bin_Regional"] = 1.0
    elif payload["shipping_distance_km"] <= 1000: payload["Dist_Bin_National"] = 1.0
    elif payload["shipping_distance_km"] <= 3000: payload["Dist_Bin_Continental"] = 1.0
    else: payload["Dist_Bin_International"] = 1.0

    # Price Quantiles
    if payload["unit_price"] < 25: payload["Price_Quantile_MediumLow"] = 1.0
    elif payload["unit_price"] < 50: payload["Price_Quantile_Medium"] = 1.0
    elif payload["unit_price"] < 100: payload["Price_Quantile_MediumHigh"] = 1.0
    else: payload["Price_Quantile_High"] = 1.0

    # 7. Checkboxes / Flags
    bool_fields = ["Is_Peak_Season", "Is_Weekend_Order", "Is_Weekend_Delivery", "holiday_period_Yes"]
    for field in bool_fields:
        payload[field] = 1.0 if raw_data.get(field) in [True, "on", 1, "1"] else 0.0

    # 8. One-Hot Encoding for categories
    cat_mappings = {
        "supplier_rating": raw_data.get("supplier_rating"),
        "shipment_mode": raw_data.get("shipment_mode"),
        "carrier_name": raw_data.get("carrier_name"),
        "region": raw_data.get("region"),
        "weather_condition": raw_data.get("weather_condition")
    }
    for prefix, val in cat_mappings.items():
        if val:
            key = f"{prefix}_{val}"
            if key in payload: payload[key] = 1.0

    # 9. Cyclical Time Features
    now = datetime.now()
    month = now.month
    day = now.weekday()
    day_of_year = now.timetuple().tm_yday
    
    payload["Month_sin"] = math.sin(2 * math.pi * month / 12)
    payload["Month_cos"] = math.cos(2 * math.pi * month / 12)
    payload["DayOfWeek_sin"] = math.sin(2 * math.pi * day / 7)
    payload["DayOfWeek_cos"] = math.cos(2 * math.pi * day / 7)
    payload["DayOfYear_sin"] = math.sin(2 * math.pi * day_of_year / 365)
    payload["DayOfYear_cos"] = math.cos(2 * math.pi * day_of_year / 365)

    df = pd.DataFrame([payload])
    df = df.reindex(columns=feature_names)
    return df.astype(np.float32)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_data = request.json
        input_df = expand_features(raw_data)
        
        # Log Active Features for Parity Check
        active = input_df.loc[:, (input_df != 0).any(axis=0)].to_dict('records')[0]
        print(f"DEBUG: Inference Feature Count: {len(input_df.columns)}")
        print(f"DEBUG: Active Features: {active}")
        
        # Predict (Class 1 = Delayed, Class 0 = On-Time)
        prob_delayed = float(model.predict_proba(input_df)[0][1])
        prediction = int(prob_delayed >= 0.5)
        
        result = {
            "on_time_probability": round(1 - prob_delayed, 4),
            "delayed_probability": round(prob_delayed, 4),
            "predicted_class": "Delayed" if prediction == 1 else "On-Time",
            "prediction": prediction
        }
        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(f"ERROR: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5001)
