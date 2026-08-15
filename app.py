"""
app.py — Smart Agri-Advisory ML Microservice

Endpoints (called by the Laravel backend via Axios/HTTP):
  GET  /api/health
  POST /api/predict/crop        {soil_ph, nitrogen, phosphorus, potassium,
                                  rainfall_mm, temperature_c, humidity_pct,
                                  climate_zone (optional, accepted but not
                                  used as a model feature -- the real Kaggle
                                  Crop Recommendation Dataset has no zone
                                  column)}
  POST /api/predict/fertilizer  {crop, soil_ph, nitrogen, phosphorus, potassium}
  POST /api/predict/price       {crop, months_ahead}
  POST /api/predict/pest        multipart/form-data with "image" file

Run:
    python app.py
Service listens on http://localhost:5000 (or override via `python app.py`
then set PORT, or edit the port at the bottom of this file).
"""
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from fertilizer import FertilizerRecommender
import price_forecast
import pest_cnn

MODEL_DIR = "models_store"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # allow the Laravel app (different host/port) to call this service

# ---- Load crop recommendation artifacts once at startup ----
crop_model = joblib.load(f"{MODEL_DIR}/crop_model.pkl")
crop_label_encoder = joblib.load(f"{MODEL_DIR}/crop_label_encoder.pkl")
crop_scaler = joblib.load(f"{MODEL_DIR}/crop_feature_scaler.pkl")
feature_cols = joblib.load(f"{MODEL_DIR}/crop_feature_cols.pkl")  # 7 features, no zone

fertilizer_engine = FertilizerRecommender()


@app.route("/api/capabilities", methods=["GET"])
def capabilities():
    """Lets the Laravel app render a 'what can the models predict?' page
    without hard-coding crop/class lists on the PHP side."""
    try:
        crop_classes = sorted(crop_label_encoder.classes_.tolist())
    except Exception:
        crop_classes = []

    pest_trained = os.path.exists(pest_cnn.MODEL_PATH)
    if pest_trained:
        try:
            with open(pest_cnn.CLASSES_PATH) as f:
                pest_classes = json.load(f)
        except Exception:
            pest_classes = pest_cnn.DEFAULT_CLASSES
    else:
        pest_classes = pest_cnn.DEFAULT_CLASSES

    return jsonify({
        "crop_recommendation": crop_classes,
        "price_forecast": price_forecast.CROPS,
        "pest_detection": {"trained": pest_trained, "classes": pest_classes},
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "agri-advisory-ml"})


@app.route("/api/predict/crop", methods=["POST"])
def predict_crop():
    data = request.get_json(force=True)
    required = ["soil_ph", "nitrogen", "phosphorus", "potassium",
                "rainfall_mm", "temperature_c", "humidity_pct"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = [data[col] for col in feature_cols]  # feature_cols order = training order
    X = crop_scaler.transform([row])
    probs = crop_model.predict_proba(X)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
        {"crop": crop_label_encoder.inverse_transform([i])[0], "confidence": round(float(probs[i]), 4)}
        for i in top5_idx
    ]
    return jsonify({"recommendations": top5, "top_crop": top5[0]["crop"]})


@app.route("/api/predict/fertilizer", methods=["POST"])
def predict_fertilizer():
    data = request.get_json(force=True)
    required = ["crop", "soil_ph", "nitrogen", "phosphorus", "potassium"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = fertilizer_engine.recommend(
        data["crop"], data["soil_ph"], data["nitrogen"], data["phosphorus"], data["potassium"]
    )
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/predict/price", methods=["POST"])
def predict_price():
    data = request.get_json(force=True)
    if "crop" not in data:
        return jsonify({"error": "Missing field: crop"}), 400
    months_ahead = int(data.get("months_ahead", 3))
    result = price_forecast.forecast(data["crop"], months_ahead)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/predict/pest", methods=["POST"])
def predict_pest():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded (field name must be 'image')"}), 400
    file = request.files["image"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    result = pest_cnn.predict_disease(path)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
