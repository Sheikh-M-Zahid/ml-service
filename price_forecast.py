"""
price_forecast.py

Forecasts short-term market price for a crop from historical price history.
Given the last WINDOW months of prices it predicts the next month.

NOTE ON LSTM: the project spec calls for an LSTM. This module trains a
lightweight MLP-on-lag-features sequence regressor by default because it
has zero heavy dependencies and trains in under a second. See
train_lstm_keras.py for a drop-in Keras/TensorFlow LSTM version of the
same interface -- app.py will use it automatically if
models_store/price_lstm_<crop>.h5 files are present.

Real per-crop market price history isn't part of the Kaggle Crop
Recommendation Dataset (it only covers soil/weather -> crop), so this
module still generates a synthetic monthly price series per crop. Swap
`generate_price_history()` for a real mandi-price CSV when available --
same column names, no other code changes needed.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

WINDOW = 6
OUT_DIR = "models_store"
DATA_PATH = "data/market_prices.csv"

CROPS = ["rice", "maize", "cotton", "jute", "banana", "mango", "coffee", "coconut"]

def generate_price_history(months=36, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    base_prices = {
        "rice": 32, "maize": 22, "cotton": 65, "jute": 40,
        "banana": 18, "mango": 55, "coffee": 220, "coconut": 35,
    }
    for crop in CROPS:
        for m in range(months):
            seasonal = 3 * np.sin(2 * np.pi * (m % 12) / 12)
            trend = 0.05 * m
            noise = rng.normal(0, 1.5)
            price = max(5, base_prices[crop] + seasonal + trend + noise)
            rows.append([crop, m, round(price, 2)])
    return pd.DataFrame(rows, columns=["crop", "month_index", "price_bdt_per_kg"])

def build_windows(series, window=WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

def train():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        df = generate_price_history()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    models = {}
    scalers = {}
    for crop in df["crop"].unique():
        series = df[df["crop"] == crop].sort_values("month_index")["price_bdt_per_kg"].values
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y = build_windows(series_scaled, WINDOW)
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=42)
        model.fit(X, y)
        models[crop] = model
        scalers[crop] = scaler

    joblib.dump(models, f"{OUT_DIR}/price_models.pkl")
    joblib.dump(scalers, f"{OUT_DIR}/price_scalers.pkl")
    joblib.dump(df, f"{OUT_DIR}/price_history.pkl")
    print(f"Trained price forecast models for: {list(models.keys())}")

def load():
    models = joblib.load(f"{OUT_DIR}/price_models.pkl")
    scalers = joblib.load(f"{OUT_DIR}/price_scalers.pkl")
    history = joblib.load(f"{OUT_DIR}/price_history.pkl")
    return models, scalers, history

def forecast(crop, months_ahead=3):
    crop = crop.strip().lower()
    models, scalers, history = load()
    if crop not in models:
        return {"error": f"No price model trained for '{crop}'"}

    series = history[history["crop"] == crop].sort_values("month_index")["price_bdt_per_kg"].values
    scaler = scalers[crop]
    model = models[crop]

    scaled = scaler.transform(series.reshape(-1, 1)).flatten()
    window = list(scaled[-WINDOW:])
    preds_scaled = []
    for _ in range(months_ahead):
        x = np.array(window[-WINDOW:]).reshape(1, -1)
        next_val = model.predict(x)[0]
        preds_scaled.append(next_val)
        window.append(next_val)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return {
        "crop": crop,
        "last_known_price_bdt_per_kg": round(float(series[-1]), 2),
        "forecast_bdt_per_kg": [round(float(p), 2) for p in preds],
        "months_ahead": months_ahead,
        "method": "sequence-regressor on last-6-months window (Keras LSTM drop-in available, see train_lstm_keras.py)",
    }

if __name__ == "__main__":
    train()
    print(forecast("rice"))
