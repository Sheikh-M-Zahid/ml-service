<div align="center">

# 🧠 ML Microservice — Smart Agri-Advisory Platform

### Flask · Random Forest · Real Kaggle Dataset — ফসল, সার ও মূল্য পূর্বাভাস API

![Flask](https://img.shields.io/badge/Flask-ML%20API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Crop%20Model-99.3%25%20Accuracy-2ECC71?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-2200%20rows%20%C2%B7%2022%20crops-orange?style=for-the-badge)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

</div>

---

## 📌 এক নজরে

এই ফোল্ডারে আছে **Flask ML মাইক্রোসার্ভিস** — Laravel backend থেকে HTTP কল পেয়ে ফসল, সার, বাজার-মূল্য এবং (পরবর্তীতে) রোগ শনাক্তকরণের পূর্বাভাস রিটার্ন করে।

> 🎉 **আপডেট:** ফসল সুপারিশ মডেল এখন প্রশিক্ষিত হয়েছে বাস্তব **Kaggle "Crop Recommendation Dataset"** দিয়ে — ২২০০ রো, ২২টি ফসল (N, P, K, temperature, humidity, ph, rainfall)। সিন্থেটিক ডেটা থেকে সরে এসে এখন এটি সত্যিকারের কৃষি ডেটার উপর কাজ করছে।

<div align="center">

| 📊 মেট্রিক | মান |
|:---:|:---:|
| **Test Accuracy** | 🎯 **৯৯.৩%** |
| **ডেটাসেট সাইজ** | 2,200 রো |
| **ফসলের সংখ্যা** | 22টি লেবেল |
| **ফিচার** | 7টি (N, P, K, তাপমাত্রা, আর্দ্রতা, pH, বৃষ্টিপাত) |

</div>

**ডেটা ফাইল:**
- `data/Crop_recommendation.csv` — মূল Kaggle ডেটাসেট
- `data/crop_data.csv` — এই সার্ভিসের কলাম-নেম অনুযায়ী রিনেম করা

> ⚠️ **নোট:** বাস্তব ডেটাসেটে কোনো `climate_zone` কলাম নেই, তাই `/api/predict/crop`-এ এখন `climate_zone` আর আবশ্যিক ফিল্ড নয় — মডেল শুধু ৭টি মাটি/আবহাওয়া ফিচার ব্যবহার করে। সার এবং price-forecast-এর ফসলের নামগুলোও এখন বাস্তব ডেটাসেটের ২২টি লেবেলের সাথে মিলিয়ে দেওয়া হয়েছে (rice, maize, chickpea, banana, mango, cotton, jute, coffee, ...) — আগের বাংলাদেশ-স্টাইল নামের বদলে। সম্পূর্ণ তালিকার জন্য দেখুন `fertilizer.py`-এর `BASE_DOSAGE` dict।

---

## 🚀 সেটআপ

```bash
cd ml-service
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🏋️ মডেল ট্রেইন করা (প্রথমবার চালানোর সময়)

```bash
python train_crop_model.py      # 🌾 বাস্তব ডেটাসেটে Random Forest ট্রেইন + সেভ
python price_forecast.py        # 📈 মূল্য পূর্বাভাস মডেল ট্রেইন + সেভ
```

## ▶️ API চালু করা

```bash
python app.py
# ➜ http://localhost:5000 (অথবা তোমার সেট করা পোর্ট)
```

---

## 🔌 API Endpoints

<div align="center">

| Method | Path | Body |
|:---:|---|---|
| 🟢 `GET` | `/api/health` | – |
| 🌾 `POST` | `/api/predict/crop` | `soil_ph, nitrogen, phosphorus, potassium, rainfall_mm, temperature_c, humidity_pct` |
| 🧪 `POST` | `/api/predict/fertilizer` | `crop, soil_ph, nitrogen, phosphorus, potassium` |
| 📈 `POST` | `/api/predict/price` | `crop, months_ahead` |
| 🍃 `POST` | `/api/predict/pest` | multipart form field `image` |

</div>

---

## 🧪 টেস্ট ডেটা

`data/test_samples.csv`-এ বাস্তব ডেটাসেট থেকে প্রতিটি ফসলের জন্য একটি করে representative রো আছে — সাইনআপ ছাড়াই farmer dashboard বা curl/Postman দিয়ে দ্রুত প্রেডিকশন যাচাই করতে এগুলো ব্যবহার করুন:

```csv
crop,soil_ph,nitrogen,phosphorus,potassium,rainfall_mm,temperature_c,humidity_pct
rice,6.5,90,42,43,202.94,20.88,82.0
maize,5.75,71,54,16,87.76,22.61,63.69
chickpea,7.49,40,72,77,88.55,17.02,16.99
...
```

**উদাহরণ curl কল:**

```bash
curl -X POST http://localhost:5000/api/predict/crop \
  -H "Content-Type: application/json" \
  -d '{
    "soil_ph": 6.5,
    "nitrogen": 90,
    "phosphorus": 42,
    "potassium": 43,
    "rainfall_mm": 202.94,
    "temperature_c": 20.88,
    "humidity_pct": 82.0
  }'
```

---

## 🔄 ডেটা ফ্লো

```
Laravel FarmerController
        │  (soil pH/N/P/K + weather)
        ▼
Flask /api/predict/crop  ──▶  🌾 Random Forest (99.3% accuracy)
        │
        ▼
ফসলের নাম রিটার্ন  ──▶  Laravel-এ crops.id-তে রিজলভ
        │
        ▼
/api/predict/fertilizer  ──▶  🧪 Rule-based + k-NN
        │
        ▼
/api/predict/price  ──▶  📈 Sequence forecast model
```

---

<div align="center">

🔗 **Live App:** [agriadvisory.app](http://agriadvisory.app)  ·  📦 **Repo:** [Agri_Advisory](https://github.com/Marjan15H/Agri_Advisory)

Made with 💚 for Bangladeshi Farmers | CSE347 Project

</div>
