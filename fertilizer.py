"""
fertilizer.py
Fertilizer recommendation = rule-based baseline dosage table (per crop)
adjusted by collaborative filtering: we look at the K nearest historical
farm profiles (by soil pH/NPK) that grew the same crop and blend their
actual applied dosage with the rule-based baseline.

Dosage values below are general agronomic approximations for the 22 crops
in the Kaggle Crop Recommendation Dataset -- swap in real regional
guidelines (e.g. from BARI) for production use.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Rule-based baseline dosage (kg/acre) -- approximate values.
BASE_DOSAGE = {
    "rice":        {"Urea": 110, "TSP": 50, "MoP": 50},
    "maize":       {"Urea": 130, "TSP": 55, "MoP": 50},
    "chickpea":    {"Urea": 20,  "TSP": 50, "MoP": 20},
    "kidneybeans": {"Urea": 25,  "TSP": 55, "MoP": 25},
    "pigeonpeas":  {"Urea": 20,  "TSP": 45, "MoP": 20},
    "mothbeans":   {"Urea": 15,  "TSP": 35, "MoP": 15},
    "mungbean":    {"Urea": 15,  "TSP": 35, "MoP": 15},
    "blackgram":   {"Urea": 15,  "TSP": 35, "MoP": 15},
    "lentil":      {"Urea": 20,  "TSP": 45, "MoP": 20},
    "pomegranate": {"Urea": 90,  "TSP": 45, "MoP": 60},
    "banana":      {"Urea": 150, "TSP": 60, "MoP": 130},
    "mango":       {"Urea": 100, "TSP": 45, "MoP": 55},
    "grapes":      {"Urea": 80,  "TSP": 55, "MoP": 70},
    "watermelon":  {"Urea": 90,  "TSP": 55, "MoP": 60},
    "muskmelon":   {"Urea": 90,  "TSP": 55, "MoP": 60},
    "apple":       {"Urea": 100, "TSP": 50, "MoP": 65},
    "orange":      {"Urea": 100, "TSP": 50, "MoP": 60},
    "papaya":      {"Urea": 110, "TSP": 55, "MoP": 70},
    "coconut":     {"Urea": 120, "TSP": 45, "MoP": 90},
    "cotton":      {"Urea": 120, "TSP": 55, "MoP": 55},
    "jute":        {"Urea": 70,  "TSP": 35, "MoP": 35},
    "coffee":      {"Urea": 100, "TSP": 45, "MoP": 65},
}

class FertilizerRecommender:
    def __init__(self, farm_profiles_csv=None):
        """farm_profiles_csv: optional historical dataset of
        [soil_ph, nitrogen, phosphorus, potassium, crop, urea_kg, tsp_kg, mop_kg].
        If not provided, a small synthetic history is generated so the
        collaborative-filtering path is demonstrable end-to-end."""
        if farm_profiles_csv:
            self.history = pd.read_csv(farm_profiles_csv)
        else:
            self.history = self._synthetic_history()
        self._fit_neighbors()

    def _synthetic_history(self, n_per_crop=30, seed=7):
        rng = np.random.default_rng(seed)
        rows = []
        for crop, dosage in BASE_DOSAGE.items():
            for _ in range(n_per_crop):
                ph = np.clip(rng.normal(6.3, 0.7), 3.5, 9.0)
                n = np.clip(rng.normal(60, 25), 0, 150)
                p = np.clip(rng.normal(50, 20), 0, 150)
                k = np.clip(rng.normal(50, 25), 0, 210)
                urea = max(0, dosage["Urea"] + rng.normal(0, 12))
                tsp = max(0, dosage["TSP"] + rng.normal(0, 8))
                mop = max(0, dosage["MoP"] + rng.normal(0, 8))
                rows.append([ph, n, p, k, crop, urea, tsp, mop])
        return pd.DataFrame(rows, columns=["soil_ph","nitrogen","phosphorus","potassium",
                                            "crop","urea_kg","tsp_kg","mop_kg"])

    def _fit_neighbors(self):
        self.models = {}
        for crop in self.history["crop"].unique():
            sub = self.history[self.history["crop"] == crop]
            X = sub[["soil_ph","nitrogen","phosphorus","potassium"]].values
            k = min(5, len(sub))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X)
            self.models[crop] = (nn, sub.reset_index(drop=True))

    def recommend(self, crop, soil_ph, nitrogen, phosphorus, potassium):
        crop_key = crop.strip().lower()
        base = BASE_DOSAGE.get(crop_key)
        if base is None:
            return {"error": f"No fertilizer rule defined for crop '{crop}'"}

        result = dict(base)
        method = "rule_based_only"

        if crop_key in self.models:
            nn, sub = self.models[crop_key]
            query = np.array([[soil_ph, nitrogen, phosphorus, potassium]])
            dist, idx = nn.kneighbors(query)
            neighbors = sub.iloc[idx[0]]
            cf_avg = {
                "Urea": neighbors["urea_kg"].mean(),
                "TSP": neighbors["tsp_kg"].mean(),
                "MoP": neighbors["mop_kg"].mean(),
            }
            result = {k: round(0.5 * base[k] + 0.5 * cf_avg[k], 1) for k in base}
            method = "rule_based + collaborative_filtering(k-NN similar farms)"

        return {
            "crop": crop,
            "recommended_dosage_kg_per_acre": result,
            "method": method,
        }
