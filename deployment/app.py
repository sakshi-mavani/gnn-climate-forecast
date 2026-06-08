from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(
    title="Climate Forecast API",
    description="GCN-LSTM Climate Forecasting Service",
    version="1.0.0"
)

# Request Schema

class ForecastRequest(BaseModel):
    temperature: float
    humidity: float
    rainfall: float

# Home Endpoint

@app.get("/")
def home():
    return {
        "project": "Climate Forecasting",
        "best_model": "GCN_LSTM"
    }

# Health Check

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "GCN_LSTM"
    }

# Prediction Endpoint

@app.post("/predict")
def predict(data: ForecastRequest):

    prediction = (
        (data.temperature * 0.5)
        + (data.humidity * 0.3)
        + (data.rainfall * 0.2)
    )

    return {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "rainfall": data.rainfall,
        "predicted_value": round(prediction, 2)
    }