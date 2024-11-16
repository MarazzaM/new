from pydantic import BaseModel
from typing import List, Dict

class DatasetUploadRequest(BaseModel):
    email: str
    password: str

class TrainingRequest(BaseModel):
    email: str
    password: str

class PredictionRequest(BaseModel):
    features: List[float]

class ModelResponse(BaseModel):
    coef: List[List[float]]
    intercept: List[float]
    scaler_mean: List[float]
    scaler_scale: List[float]