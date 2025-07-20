import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional, List
from sklearn.preprocessing import MinMaxScaler

MODEL_ASSETS_DIR = "model_assets"
MODEL_PATH = os.path.join(MODEL_ASSETS_DIR, 'random_forest_model.joblib')
ENCODERS_PATH = os.path.join(MODEL_ASSETS_DIR, 'label_encoders_dict.joblib')
SCALER_PATH = os.path.join(MODEL_ASSETS_DIR, 'minmax_scaler.joblib') 

loaded_model = None
loaded_encoders = None
loaded_scaler = None 

EXPECTED_FEATURE_ORDER = [
    'Experience',
    'Age',
    'Education_encoded',
    'Location_encoded',
    'Job_Title_encoded',
    'Gender_encoded'
]

app = FastAPI(
    title="Employee Salary Prediction API",
    description="Predicts employee salaries using a pre-trained Random Forest Regressor model."
)

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

    # 1. Process Categorical Features using Label Encoders
    for feature_name, encoder in loaded_encoders.items():
        raw_value = input_data.get(feature_name)
        if raw_value is None:
            raise HTTPException(status_code=400, detail=f"Missing categorical feature: {feature_name}")
        try:
            encoded_value = encoder.transform([raw_value])[0]
            processed_features[f'{feature_name}_encoded'] = encoded_value
        except ValueError:
            print(f"Warning: Unseen category '{raw_value}' for '{feature_name}'. Assigning -1.")
            processed_features[f'{feature_name}_encoded'] = -1
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding {feature_name}: {e}")

    # 2. Add Numerical Features directly
    numerical_features_raw_names = ['Experience', 'Age'] # Original names for numerical features
    for num_feature in numerical_features_raw_names:
        value = input_data.get(num_feature)
        if value is None:
            raise HTTPException(status_code=400, detail=f"Missing numerical feature: {num_feature}")
        processed_features[num_feature] = value

    # 3. Create a Pandas DataFrame for prediction, ensuring correct column order
    try:
        input_df = pd.DataFrame([processed_features])
        # Reorder columns to match the training data's EXPECTED_FEATURE_ORDER
        # This is crucial for the scaler and the model
        input_df = input_df[EXPECTED_FEATURE_ORDER]
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Internal error: Feature mismatch. Missing expected column: {e}. Check EXPECTED_FEATURE_ORDER.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing input data for prediction: {e}")

    # 4. Apply Scaling using the loaded MinMaxScaler
    try:
        # The scaler expects a 2D array, so input_df is correct
        input_df_scaled = loaded_scaler.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying scaler: {e}")

    # 5. Make the prediction
    try:
        prediction = loaded_model.predict(input_df_scaled)[0]
        return {"predicted_salary_inr": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Employee Salary Prediction API! Use /predict_salary to get predictions."}

