from fastapi import FastAPI
from pydantic import BaseModel  #validates input data format
import numpy as np
import joblib

app = FastAPI()

# -------------------------
# LOAD MODEL + SCALER
# -------------------------
model = joblib.load(r"C:\Users\shrey\OneDrive\Documents\Desktop\Mlops\STREAMSHIELD_AI\models\random_forest.pkl")

try:
    scaler = joblib.load(r"C:\Users\shrey\OneDrive\Documents\Desktop\Mlops\STREAMSHIELD_AI\models\scaler.pkl")
except:
    scaler = None

# -------------------------
# INPUT SCHEMA
# -------------------------
class InputData(BaseModel):
    features: list
    model: str = "Model A"

# -------------------------
# ROOT ENDPOINT (for status check)
# -------------------------
@app.get("/")
def home():
    return {"status": "API running"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1) #rehsapes to 2d

    # Apply scaling
    if scaler:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]

    # Example mapping
    label_map = {
        17: "Benign",
        18: "Benign"
    }

    result = label_map.get(int(prediction), "Attack")

    return {
        "prediction": int(prediction),
        "result": result
    }

# -------------------------
# METRICS (CONFUSION MATRIX)
# -------------------------
@app.get("/metrics")
def metrics():
    # Demo matrix
    cm = [[50, 2],
          [3, 45]]

    return {
        "confusion_matrix": cm,
        "accuracy": 0.95
    }