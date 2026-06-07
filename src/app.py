from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Machine Failure Detection API", version="1.0")

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class MachineData(BaseModel):
    Type: int
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float
    Torque_x_Toolwear: float
    Temp_difference: float

@app.get("/")
def home():
    return {"message": "Machine Failure Detection API is running"}

@app.post("/predict")
def predict(data: MachineData):
    features = np.array([[
        data.Type,
        data.Air_temperature_K,
        data.Process_temperature_K,
        data.Rotational_speed_rpm,
        data.Torque_Nm,
        data.Tool_wear_min,
        data.Torque_x_Toolwear,
        data.Temp_difference
    ]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "result": "FAILURE DETECTED" if prediction == 1 else "Normal Operation",
        "failure_probability": round(float(probability), 4)
    }
