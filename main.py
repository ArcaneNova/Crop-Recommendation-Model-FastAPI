from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os
import io
from PIL import Image
import tensorflow as tf
from typing import List, Optional

# Load the crop recommendation model, scaler, and crop mapping
crop_model_path = "crop_recommendation_model.pkl"
scaler_path = "scaler.pkl"
mapping_path = "crop_mapping.pkl"

# Check if crop model files exist
if not os.path.exists(crop_model_path) or not os.path.exists(scaler_path) or not os.path.exists(mapping_path):
    raise FileNotFoundError("Crop model files not found. Make sure the .pkl files are in the correct location.")

# Load the crop recommendation model, scaler, and crop mapping
crop_model = joblib.load(crop_model_path)
scaler = joblib.load(scaler_path)
crop_mapping = joblib.load(mapping_path)

# Create FastAPI app
app = FastAPI(
    title="Crop Recommendation API",
    description="API for crop recommendation",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define input data model for crop recommendation
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define response model for crop recommendation
class CropResponse(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Crop Recommendation API"}

@app.post("/predict/crop")
async def predict_crop(crop_input: CropInput):
    try:
        # Extract the input features
        features = np.array([[
            crop_input.N,
            crop_input.P,
            crop_input.K,
            crop_input.temperature,
            crop_input.humidity,
            crop_input.ph,
            crop_input.rainfall
        ]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = crop_model.predict(scaled_features)
        
        # Get the crop name
        crop_name = crop_mapping[prediction[0]]
        
        return {
            "prediction": crop_name,
            "confidence": 0.95  # Placeholder confidence value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crops")
def get_available_crops():
    """Return a list of all possible crops that can be recommended"""
    return {"available_crops": list(crop_mapping.values())}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 