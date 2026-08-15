"""
train_crop_model.py
Trains a Random Forest classifier on the real Kaggle "Crop Recommendation
Dataset" (2200 rows, 22 crops: N, P, K, temperature, humidity, ph, rainfall).

Run:  python train_crop_model.py
Outputs: models_store/crop_model.pkl, models_store/crop_label_encoder.pkl,
         models_store/crop_feature_scaler.pkl, models_store/crop_feature_cols.pkl
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/crop_data.csv"
OUT_DIR = "models_store"
os.makedirs(OUT_DIR, exist_ok=True)

# NOTE: this real dataset has no climate-zone column, so the model is
# trained on the 7 soil/weather features only. climate_zone is still
# accepted by the API for logging/display but is not used as a feature.
FEATURE_COLS = ["soil_ph", "nitrogen", "phosphorus", "potassium",
                 "rainfall_mm", "temperature_c", "humidity_pct"]

def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].values
    y_raw = df["crop"].values

    crop_encoder = LabelEncoder()
    y = crop_encoder.fit_transform(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=14, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=crop_encoder.classes_, zero_division=0))

    joblib.dump(clf, f"{OUT_DIR}/crop_model.pkl")
    joblib.dump(crop_encoder, f"{OUT_DIR}/crop_label_encoder.pkl")
    joblib.dump(scaler, f"{OUT_DIR}/crop_feature_scaler.pkl")
    joblib.dump(FEATURE_COLS, f"{OUT_DIR}/crop_feature_cols.pkl")
    print("Saved model artifacts to models_store/")

if __name__ == "__main__":
    main()
