# krishi_mitra_nn.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import joblib
import logging

# -------------------------
# Logging setup
# -------------------------
logging.basicConfig(
    filename="crop_predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Krishi Mitra AI Server")

# -------------------------
# Load Disease Model
# -------------------------
disease_model_path = os.path.join("model", "plant_model.h5")
if not os.path.exists(disease_model_path):
    raise FileNotFoundError(f"Disease model file not found at {disease_model_path}")

disease_model = tf.keras.models.load_model(disease_model_path)
diseases = ["Healthy", "Leaf Blight", "Powdery Mildew", "Rust"]

# -------------------------
# Load Crop Neural Network Model
# -------------------------
crop_model_path = os.path.join("model", "crop_nn_model.h5")
if not os.path.exists(crop_model_path):
    raise FileNotFoundError(f"Crop neural network model not found at {crop_model_path}")

crop_model = tf.keras.models.load_model(crop_model_path)
le = joblib.load(os.path.join("model", "crop_label_encoder.pkl"))
X_min, X_max = joblib.load(os.path.join("model", "crop_minmax.pkl"))

def normalize_features(features: np.ndarray) -> np.ndarray:
    return (features - X_min) / (X_max - X_min)

# -------------------------
# Crop Input Schema
# -------------------------
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# -------------------------
# Disease Prediction API
# -------------------------
@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = image.resize((224, 224))
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = disease_model.predict(img)
        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        return {"disease": diseases[index], "confidence": round(confidence, 2)}
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Crop Recommendation API (Neural Network)
# -------------------------
@app.post("/predict-crop")
def predict_crop(data: CropInput):
    try:
        features = np.array([[data.N, data.P, data.K, data.temperature,
                              data.humidity, data.ph, data.rainfall]])
        features_norm = normalize_features(features)

        # Neural network prediction
        prediction_probs = crop_model.predict(features_norm)[0]
        index = np.argmax(prediction_probs)
        predicted_crop = le.inverse_transform([index])[0]

        # convert numpy.float32 to float for JSON
        prob_dict = {crop: float(round(prediction_probs[i]*100, 2)) for i, crop in enumerate(le.classes_)}

        return {"recommended_crop": predicted_crop, "probabilities": prob_dict}

    except Exception as e:
        return {"error": str(e)}
# Health Check API
# -------------------------
@app.get("/")
def home():
    return {"message": "Krishi Mitra AI Server Running"}