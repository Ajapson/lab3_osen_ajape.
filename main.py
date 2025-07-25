# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
try:
    model, label_encoder, feature_columns = joblib.load("app/data/model.json")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Define enums for input validation
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

# Define input schema
class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Create FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Penguin Species Classifier API!"}

@app.post("/predict")
def predict_penguin(data: PenguinFeatures):
    try:
        logger.info(f"Input received: {data}")

        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # One-hot encode the input
        input_df = pd.get_dummies(input_df)

        # Ensure the input matches training columns
        for col in feature_columns:
            if col not in input_df:
                input_df[col] = 0
        input_df = input_df[feature_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        species = label_encoder.inverse_transform([prediction])[0]

        logger.info(f"Prediction successful: {species}")
        return {"predicted_species": species}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

