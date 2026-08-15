"""
generate_data.py
Generates a synthetic dataset that mimics Bangladesh Agriculture Research
Institute (BARI) style soil/weather -> crop suitability data, since a live
BARI dataset is not bundled with this repo. Replace this with the real
BARI dataset (CSV with the same columns) when available -- the training
script does not need to change.

Columns: soil_ph, nitrogen, phosphorus, potassium, rainfall_mm,
         temperature_c, humidity_pct, climate_zone, crop
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# Approximate agro-climatic ranges for common Bangladesh crops.
# (ph, N, P, K, rainfall_mm, temp_c, humidity_pct) as (mean, std)
CROP_PROFILES = {
    "Rice (Boro)":     dict(ph=(6.0, 0.4), n=(80, 10), p=(35, 6),  k=(35, 6),  rain=(180, 40), temp=(27, 3), hum=(75, 8)),
    "Rice (Aman)":     dict(ph=(6.2, 0.4), n=(70, 10), p=(30, 6),  k=(30, 6),  rain=(300, 60), temp=(28, 2), hum=(80, 6)),
    "Wheat":           dict(ph=(6.8, 0.4), n=(100, 12), p=(50, 8), k=(40, 6), rain=(60, 20),  temp=(20, 3), hum=(55, 8)),
    "Jute":            dict(ph=(6.5, 0.4), n=(60, 10), p=(25, 5),  k=(30, 5),  rain=(250, 50), temp=(30, 2), hum=(78, 6)),
    "Potato":          dict(ph=(5.8, 0.3), n=(120, 15), p=(60, 10),k=(90, 12), rain=(50, 15),  temp=(18, 3), hum=(65, 8)),
    "Maize":           dict(ph=(6.3, 0.4), n=(110, 12), p=(45, 8), k=(40, 8),  rain=(120, 30), temp=(26, 3), hum=(60, 8)),
    "Lentil (Masur)":  dict(ph=(6.7, 0.4), n=(20, 6),  p=(40, 8),  k=(20, 5),  rain=(30, 12),  temp=(21, 3), hum=(55, 8)),
    "Mustard":         dict(ph=(6.5, 0.4), n=(60, 10), p=(30, 6),  k=(25, 5),  rain=(25, 10),  temp=(19, 3), hum=(58, 8)),
    "Sugarcane":       dict(ph=(6.5, 0.4), n=(150, 15),p=(60, 10), k=(120, 15),rain=(220, 40), temp=(29, 2), hum=(75, 6)),
    "Tea":             dict(ph=(5.0, 0.3), n=(90, 10), p=(30, 6),  k=(50, 8),  rain=(350, 60), temp=(24, 3), hum=(82, 5)),
    "Tomato":          dict(ph=(6.2, 0.3), n=(100, 12),p=(55, 8),  k=(80, 10), rain=(60, 20),  temp=(23, 3), hum=(62, 8)),
    "Onion":           dict(ph=(6.4, 0.3), n=(90, 10), p=(45, 8),  k=(60, 8),  rain=(40, 15),  temp=(22, 3), hum=(58, 8)),
    "Chili":           dict(ph=(6.3, 0.3), n=(80, 10), p=(40, 8),  k=(50, 8),  rain=(90, 25),  temp=(27, 3), hum=(68, 8)),
    "Brinjal":         dict(ph=(6.0, 0.3), n=(85, 10), p=(40, 8),  k=(55, 8),  rain=(100, 25), temp=(28, 3), hum=(70, 8)),
    "Cotton":          dict(ph=(6.6, 0.4), n=(100, 12),p=(45, 8),  k=(45, 8),  rain=(80, 25),  temp=(29, 2), hum=(60, 8)),
}

ZONES = ["North-West (Barind)", "North-East (Haor)", "Central (Ganges Plain)",
         "South-West (Coastal)", "South-East (Hilly)", "North (Brahmaputra Char)"]

def sample_row(crop, profile, rng):
    ph = np.clip(rng.normal(*profile["ph"]), 3.5, 9.0)
    n = np.clip(rng.normal(*profile["n"]), 0, 300)
    p = np.clip(rng.normal(*profile["p"]), 0, 150)
    k = np.clip(rng.normal(*profile["k"]), 0, 250)
    rain = np.clip(rng.normal(*profile["rain"]), 0, 600)
    temp = np.clip(rng.normal(*profile["temp"]), 5, 42)
    hum = np.clip(rng.normal(*profile["hum"]), 10, 100)
    zone = rng.choice(ZONES)
    return [round(ph,2), round(n,1), round(p,1), round(k,1),
            round(rain,1), round(temp,1), round(hum,1), zone, crop]

def generate(n_per_crop=250, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(n_per_crop):
            rows.append(sample_row(crop, profile, rng))
    df = pd.DataFrame(rows, columns=["soil_ph","nitrogen","phosphorus","potassium",
                                      "rainfall_mm","temperature_c","humidity_pct",
                                      "climate_zone","crop"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    df = generate()
    df.to_csv("data/crop_data.csv", index=False)
    print(f"Generated {len(df)} rows -> data/crop_data.csv")
    print(df.head())
