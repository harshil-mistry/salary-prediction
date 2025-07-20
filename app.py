import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from typing import Optional, List
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

MODEL_ASSETS_DIR = "model_assets"
MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, 'random_forest_model.joblib')
ENCODERS_PATH = os.path.join(MODEL_ASSETS_DIR, 'label_encoders_dict.joblib')
SCALER_PATH = os.path.join(MODEL_ASSETS_DIR, 'minmax_scaler.joblib') # Path for the scaler

loaded_model = None
loaded_encoders = None
loaded_scaler = None 

EXPECTED_FEATURE_ORDER = [
    'Education',
    'Experience',
    'Location',
    'Job_Title',
    'Age',
    'Gender'
]

app = FastAPI(
    title="Employee Salary Prediction API",
    description="Predicts employee salaries using a custom-trained Random Forest Regressor model.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

class EmployeeFeatures(BaseModel):
    Education: str
    Experience: int
    Location: str
    Job_Title: str
    Age: int
    Gender: str

@app.on_event("startup")
async def load_assets_on_startup():
    
    global loaded_model, loaded_encoders, loaded_scaler

    print(f"Attempting to load model from: {MODEL_PATH}")
    print(f"Attempting to load encoders from: {ENCODERS_PATH}")
    print(f"Attempting to load scaler from: {SCALER_PATH}")

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_encoders = joblib.load(ENCODERS_PATH)
        loaded_scaler = joblib.load(SCALER_PATH)

        print("Model, Encoders, and Scaler loaded successfully!")
    except FileNotFoundError as e:
        print(f"ERROR: Asset file not found. Please ensure '{MODEL_ASSETS_DIR}' exists and contains all required files.")
        print(f"Missing file: {e.filename}")
        raise HTTPException(status_code=500, detail=f"Server setup error: Missing model asset file ({e.filename}).")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during asset loading: {e}")
        raise HTTPException(status_code=500, detail=f"Server setup error: Failed to load model assets. ({e})")

@app.post("/predict_salary")
async def predict_salary(employee_data: EmployeeFeatures):
    
    if loaded_model is None or loaded_encoders is None or loaded_scaler is None:
        raise HTTPException(status_code=500, detail="Model assets not loaded. Please check server logs.")

    input_data = employee_data.dict()
    processed_features = {}

    categorical_features_raw_names = ['Education', 'Location', 'Job_Title', 'Gender']
    for feature_name in categorical_features_raw_names:
        raw_value = input_data.get(feature_name)
        if raw_value is None:
            raise HTTPException(status_code=400, detail=f"Missing categorical feature: {feature_name}")
        
        encoder = loaded_encoders.get(feature_name)
        if encoder is None:
            raise HTTPException(status_code=500, detail=f"Encoder for '{feature_name}' not found. Check 'label_encoders_dict.joblib'.")

        try:
            encoded_value = encoder.transform([raw_value])[0]
            processed_features[f'{feature_name}'] = encoded_value
        except ValueError:
            print(f"Warning: Unseen category '{raw_value}' for '{feature_name}'. Assigning -1.")
            processed_features[f'{feature_name}'] = -1
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding {feature_name}: {e}")

    numerical_features_raw_names = ['Experience', 'Age']
    for num_feature in numerical_features_raw_names:
        value = input_data.get(num_feature)
        if value is None:
            raise HTTPException(status_code=400, detail=f"Missing numerical feature: {num_feature}")
        processed_features[num_feature] = value

    try:
        input_df = pd.DataFrame([processed_features])
        input_df = input_df[EXPECTED_FEATURE_ORDER]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Internal error: Feature mismatch. Missing expected column: {e}. Check EXPECTED_FEATURE_ORDER.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing input data for prediction: {e}")

    try:
        input_df_scaled = loaded_scaler.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying scaler: {e}")

    try:
        prediction = loaded_model.predict(input_df_scaled)[0]
        return {"predicted_salary_inr": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_prediction_form(request: Request):
    # Render the index.html template
    return templates.TemplateResponse("index.html", {"request": request})