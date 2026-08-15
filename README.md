# Agri-Advisory ML Microservice (Flask)

Crop recommendation is now trained on the **real Kaggle "Crop
Recommendation Dataset"** (2200 rows, 22 crops: N, P, K, temperature,
humidity, ph, rainfall) — see `data/Crop_recommendation.csv` (original)
and `data/crop_data.csv` (renamed to this service's column names).

**Test accuracy: 99.3%** (see `train_crop_model.py` output).

Note: this real dataset has no climate-zone column, so `climate_zone` is
no longer a required field on `/api/predict/crop` — the model uses only
the 7 soil/weather features. Fertilizer and price-forecast crop names now
match the real dataset's 22 crop labels (rice, maize, chickpea, banana,
mango, cotton, jute, coffee, ...) instead of the earlier Bangladesh-style
names — see `fertilizer.py`'s `BASE_DOSAGE` dict for the full list.

## Setup

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the models (first run only)

```bash
python train_crop_model.py      # trains + saves Random Forest on the real dataset
python price_forecast.py        # trains + saves price forecast models
```

## Run the API

```bash
python app.py
# -> http://localhost:5000 (or the port you set)
```

## Endpoints

| Method | Path                     | Body                                                                        |
|--------|--------------------------|-------------------------------------------------------------------------------|
| GET    | /api/health              | –                                                                              |
| POST   | /api/predict/crop        | `soil_ph, nitrogen, phosphorus, potassium, rainfall_mm, temperature_c, humidity_pct` |
| POST   | /api/predict/fertilizer  | `crop, soil_ph, nitrogen, phosphorus, potassium`                              |
| POST   | /api/predict/price       | `crop, months_ahead`                                                          |
| POST   | /api/predict/pest        | multipart form field `image`                                                  |

## Test data

`data/test_samples.csv` has one representative row per crop straight from
the real dataset — use these in the farmer dashboard or with curl/Postman
to sanity-check predictions without needing to sign up:

```
crop,soil_ph,nitrogen,phosphorus,potassium,rainfall_mm,temperature_c,humidity_pct
rice,6.5,90,42,43,202.94,20.88,82.0
maize,5.75,71,54,16,87.76,22.61,63.69
chickpea,7.49,40,72,77,88.55,17.02,16.99
...
```
