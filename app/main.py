from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import syft as sy
import numpy as np
from typing import Dict, List

from app.models.schemas import (
    DatasetUploadRequest,
    TrainingRequest,
    PredictionRequest,
    ModelResponse
)
from app.services.encryption import create_context, encrypt_data
from app.services.ml import train_logistic_regression, prepare_model
from app.utils.data import load_breast_cancer_data

app = FastAPI(title="Secure ML API")

# Global variables to store necessary objects
data_site = None
client = None
ctx = None
model = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global data_site, client, ctx
    # Initialize TenSEAL context
    ctx = create_context()
    # Initialize Syft connection
    data_site = sy.orchestra.launch(name="cancer-research-centre", reset=True)

@app.post("/upload-dataset")
async def upload_dataset(request: DatasetUploadRequest):
    global client
    try:
        client = data_site.login(email=request.email, password=request.password)
        X, y = load_breast_cancer_data()
        
        # Create and upload dataset (similar to notebook cells 8-10)
        features_asset = sy.Asset(
            name="Breast Cancer Data: Features",
            data=X,
            mock=X + np.random.normal(0, 1, X.shape)
        )
        
        targets_asset = sy.Asset(
            name="Breast Cancer Data: Targets",
            data=y,
            mock=y.sample(frac=1).reset_index(drop=True)
        )
        
        dataset = sy.Dataset(
            name="Breast Cancer Biomarker",
            description="Breast cancer dataset with features and target labels.",
            summary="Predict whether the cancer is benign or malignant."
        )
        
        dataset.add_asset(features_asset)
        dataset.add_asset(targets_asset)
        client.upload_dataset(dataset=dataset)
        
        return {"message": "Dataset uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(request: TrainingRequest):
    global client, model, scaler
    try:
        client = data_site.login(email=request.email, password=request.password)
        bc_dataset = client.datasets["Breast Cancer Biomarker"]
        features, targets = bc_dataset.assets
        
        # Train model using encrypted data
        model_data = train_logistic_regression(features.mock, targets.mock)
        model, scaler = prepare_model(model_data)
        
        return ModelResponse(
            coef=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    try:
        # Convert input data to correct format and make prediction
        X_new = np.array(request.features).reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)
        probability = model.predict_proba(X_new_scaled)[0].tolist()
        
        return {
            "prediction": int(prediction[0]),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)