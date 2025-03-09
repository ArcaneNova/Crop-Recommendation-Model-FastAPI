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

# Load the plant disease model if it exists
plant_disease_model_path = "Modelplanit_ACC97.23.h5"  # Try using the alternative model
plant_disease_model = None
disease_labels = None
plant_disease_available = False

# Define disease labels (these should match your training data)
disease_labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Define a custom InputLayer class to handle the batch_shape parameter
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, input_shape=None, batch_size=None, dtype=None, sparse=None, 
                 name=None, ragged=None, type_spec=None, batch_shape=None, **kwargs):
        # Handle the batch_shape parameter by converting it to input_shape
        if batch_shape is not None and input_shape is None:
            input_shape = batch_shape[1:]  # Remove the batch dimension
        
        super().__init__(
            input_shape=input_shape, 
            batch_size=batch_size, 
            dtype=dtype, 
            sparse=sparse, 
            name=name, 
            ragged=ragged, 
            type_spec=type_spec, 
            **kwargs
        )

# Try to load the plant disease model
if os.path.exists(plant_disease_model_path):
    try:
        print(f"Found plant disease model at {plant_disease_model_path}")
        print(f"File size: {os.path.getsize(plant_disease_model_path) / (1024 * 1024):.2f} MB")
        
        # Register the custom layer
        custom_objects = {'InputLayer': CustomInputLayer}
        
        # Load the model with the custom objects
        plant_disease_model = tf.keras.models.load_model(
            plant_disease_model_path,
            compile=False,
            custom_objects=custom_objects
        )
        
        # Test the model with a dummy input to verify it works
        dummy_input = np.zeros((1, 224, 224, 3))
        try:
            _ = plant_disease_model.predict(dummy_input)
            plant_disease_available = True
            print("Plant disease model loaded and tested successfully!")
        except Exception as test_error:
            print(f"Error testing plant disease model: {str(test_error)}")
            print("Model loaded but failed testing.")
    except Exception as e:
        print(f"Error loading plant disease model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("Plant disease detection will not be available.")
else:
    print(f"Plant disease model file '{plant_disease_model_path}' not found.")
    print("Plant disease detection will not be available.")

# Create FastAPI app
app = FastAPI(
    title="Agricultural AI API",
    description="API for crop recommendation and plant disease detection",
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
class SoilData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define response model for crop recommendation
class CropResponse(BaseModel):
    recommended_crop: str

# Define response model for plant disease prediction
class DiseasePrediction(BaseModel):
    disease: str
    confidence: float
    health_score: float
    prevention_tips: Optional[str] = None

# Define response model for API status
class ApiStatus(BaseModel):
    crop_recommendation: bool
    plant_disease_detection: bool
    message: str

# Disease prevention tips dictionary
disease_prevention = {
    "Apple___Apple_scab": "Avoid overhead watering, prune infected branches, and use fungicides like copper-based sprays.",
    "Apple___Black_rot": "Remove infected leaves, apply fungicides, and ensure proper air circulation.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper plants, use resistant apple varieties, and apply fungicides.",
    "Blueberry___healthy": "Maintain good soil drainage, prune old canes, and use disease-resistant varieties.",
    "Cherry_(including_sour)___Powdery_mildew": "Ensure good air circulation, apply sulfur-based fungicides, and avoid excessive nitrogen fertilizer.",
    "Corn___Common_rust": "Use resistant seed varieties, rotate crops, and apply fungicides if necessary.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, remove infected leaves, and use resistant hybrids.",
    "Grape___Black_rot": "Prune and destroy infected vines, use sulfur-based fungicides, and avoid wet foliage.",
    "Grape___Esca_(Black_Measles)": "Prune affected areas, improve soil health, and apply fungicides if needed.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Remove infected leaves, apply copper-based fungicides, and maintain good vineyard sanitation.",
    "Orange___Haunglongbing_(Citrus_greening)": "Control psyllid insects, remove infected trees, and use resistant rootstocks.",
    "Peach___Bacterial_spot": "Apply copper-based sprays, avoid overhead irrigation, and use resistant varieties.",
    "Pepper,_bell___Bacterial_spot": "Use disease-free seeds, apply copper fungicides, and avoid handling plants when wet.",
    "Potato___Early_blight": "Rotate crops, remove infected debris, and apply fungicides like chlorothalonil.",
    "Potato___Late_blight": "Ensure good air circulation, apply fungicides, and avoid excessive moisture.",
    "Strawberry___Leaf_scorch": "Remove infected leaves, avoid overhead watering, and use resistant varieties.",
    "Tomato___Bacterial_spot": "Use disease-free seeds, remove infected plants, and apply copper fungicides.",
    "Tomato___Early_blight": "Apply fungicides, rotate crops, and prune lower leaves to improve airflow.",
    "Tomato___Late_blight": "Destroy infected plants, avoid excessive watering, and apply fungicides like metalaxyl.",
    "Tomato___Leaf_Mold": "Increase air circulation, reduce humidity, and apply copper-based fungicides.",
    "Tomato___Septoria_leaf_spot": "Mulch around plants, remove infected leaves, and use fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap, release predatory mites, and avoid dry conditions.",
    "Tomato___Target_Spot": "Apply fungicides, improve air circulation, and remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies, use resistant varieties, and apply insecticidal soap.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants, disinfect tools, and avoid handling plants excessively.",
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Agricultural AI API"}

@app.get("/status", response_model=ApiStatus)
def get_api_status():
    """Return the status of the API and available features"""
    message = "All features are available." if plant_disease_available else "Plant disease detection is not available. Crop recommendation is working."
    return {
        "crop_recommendation": True,
        "plant_disease_detection": plant_disease_available,
        "message": message
    }

@app.post("/predict", response_model=CropResponse)
def predict_crop(data: SoilData):
    try:
        # Convert input data to numpy array
        input_data = np.array([[
            data.N, 
            data.P, 
            data.K, 
            data.temperature, 
            data.humidity, 
            data.ph, 
            data.rainfall
        ]])
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = crop_model.predict(input_scaled)
        
        # Get the crop name from the mapping
        recommended_crop = crop_mapping[int(prediction[0])]
        
        return {"recommended_crop": recommended_crop}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/crops")
def get_available_crops():
    """Return a list of all possible crops that can be recommended"""
    return {"available_crops": list(crop_mapping.values())}

@app.post("/predict-disease", response_model=DiseasePrediction)
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded image
    """
    if not plant_disease_available:
        raise HTTPException(
            status_code=503, 
            detail="Plant disease detection is not available. The model file 'Modelplanit_ACC97.23.h5' is missing or could not be loaded."
        )
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Check if image is RGB, if not convert it
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize and preprocess the image
        image = image.resize((224, 224))  # Resize to match model input size
        image_array = np.asarray(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = plant_disease_model.predict(image_array)
        pred_index = np.argmax(predictions[0])
        
        # Ensure pred_index is within the valid range
        if pred_index >= len(disease_labels):
            raise HTTPException(
                status_code=500,
                detail=f"Prediction index {pred_index} is out of range for disease labels (0-{len(disease_labels)-1})"
            )
            
        confidence = float(predictions[0][pred_index])
        disease = disease_labels[pred_index]
        
        # Calculate health score
        health_score = 100.0 if "healthy" in disease.lower() else (1.0 - confidence) * 100.0
        
        # Get prevention tips
        prevention_tip = disease_prevention.get(disease, "No specific prevention tips available.")
        
        return {
            "disease": disease,
            "confidence": confidence,
            "health_score": health_score,
            "prevention_tips": prevention_tip
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Disease prediction error: {str(e)}")

@app.get("/diseases")
def get_available_diseases():
    """Return a list of all possible plant diseases that can be detected"""
    if not plant_disease_available:
        raise HTTPException(
            status_code=503, 
            detail="Plant disease detection is not available. The model file 'Modelplanit_ACC97.23.h5' is missing or could not be loaded."
        )
    
    return {"available_diseases": disease_labels}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 