"""
pest_cnn.py  (OPTIONAL ADVANCED FEATURE)

Crop leaf disease detection via CNN image classification. This needs a
real labeled image dataset to train (none is bundled -- the Kaggle Crop
Recommendation Dataset used elsewhere in this service is soil/weather
data only, not images). Ships as:

  1. A working Keras/TensorFlow CNN architecture + training script.
  2. A Flask-ready `predict_disease(image_path)` function that loads the
     trained model if present, else returns a clear "not trained yet"
     response instead of a fake prediction.

To activate real predictions:
    pip install tensorflow --break-system-packages
    # put labeled images under data/pest_images/<class_name>/*.jpg
    python pest_cnn.py --train
"""
import os
import json

OUT_DIR = "models_store"
MODEL_PATH = f"{OUT_DIR}/pest_cnn.h5"
CLASSES_PATH = f"{OUT_DIR}/pest_classes.json"
IMG_SIZE = (128, 128)

DEFAULT_CLASSES = [
    "Rice_Blast", "Rice_Brown_Spot", "Rice_Healthy",
    "Maize_Leaf_Blight", "Maize_Healthy",
    "Cotton_Leaf_Curl", "Cotton_Healthy",
]

def build_model(num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def train_pest_cnn(data_dir="data/pest_images", epochs=15):
    from tensorflow import keras
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"'{data_dir}' not found. Add labeled images as "
            f"'{data_dir}/<class_name>/*.jpg' before training."
        )
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=32)
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=32)

    class_names = train_ds.class_names
    model = build_model(len(class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save(MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Saved model -> {MODEL_PATH}, classes -> {CLASSES_PATH}")

def predict_disease(image_path):
    if not os.path.exists(MODEL_PATH):
        return {
            "status": "model_not_trained",
            "message": (
                "The pest/disease CNN has not been trained yet. Collect "
                "labeled leaf images under data/pest_images/<class>/*.jpg "
                "and run `python pest_cnn.py --train`."
            ),
            "candidate_classes": DEFAULT_CLASSES,
        }
    from tensorflow import keras
    import numpy as np

    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)

    img = keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return {
        "status": "ok",
        "predicted_class": class_names[top_idx],
        "confidence": round(float(probs[top_idx]), 4),
        "all_probabilities": {c: round(float(p), 4) for c, p in zip(class_names, probs)},
    }

if __name__ == "__main__":
    import sys
    if "--train" in sys.argv:
        train_pest_cnn()
    else:
        print(predict_disease("sample.jpg"))
