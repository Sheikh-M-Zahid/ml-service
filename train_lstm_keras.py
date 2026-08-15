"""
train_lstm_keras.py  (OPTIONAL — heavier dependency)

A literal LSTM implementation matching the project spec, kept separate from
price_forecast.py so the core service has no hard TensorFlow dependency.

    pip install tensorflow --break-system-packages
    python train_lstm_keras.py

This will save models_store/price_lstm_<crop>.h5 files. Update app.py's
price prediction handler to load these instead of price_models.pkl if you
want the real LSTM in production — the input/output contract (a WINDOW=6
month rolling window -> next month's price) is identical, so no other
code needs to change.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW = 6
DATA_PATH = "data/market_prices.csv"
OUT_DIR = "models_store"

def build_windows(series, window=WINDOW):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

def main():
    from tensorflow import keras
    from tensorflow.keras import layers

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        from price_forecast import generate_price_history
        os.makedirs("data", exist_ok=True)
        generate_price_history().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)

    for crop in df["crop"].unique():
        series = df[df["crop"] == crop].sort_values("month_index")["price_bdt_per_kg"].values
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y = build_windows(series_scaled, WINDOW)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = keras.Sequential([
            layers.Input(shape=(WINDOW, 1)),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=100, verbose=0)

        safe_name = crop.replace(" ", "_").replace("(", "").replace(")", "")
        model.save(f"{OUT_DIR}/price_lstm_{safe_name}.h5")
        print(f"Saved LSTM model for {crop}")

if __name__ == "__main__":
    main()
