# Secure Machine Learning API Documentation

## Overview
This project implements a secure machine learning API for breast cancer prediction using FastAPI, homomorphic encryption, and privacy-preserving techniques. The system utilizes PySyft for secure data handling and TenSEAL for homomorphic encryption.

## Architecture

### Core Components
1. **FastAPI Application** - Main API server handling requests and responses
2. **PySyft Integration** - Secure data handling and privacy-preserving computations
3. **TenSEAL** - Homomorphic encryption implementation
4. **Machine Learning Pipeline** - Logistic regression model for cancer prediction

## API Endpoints

The API documentation is available via Swagger UI at `/docs` when running the server.

### 1. Upload Dataset (`POST /upload-dataset`)
Uploads and encrypts breast cancer dataset to the secure data site.

**Request Body:**
```json
{
    "email": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "message": "Dataset uploaded successfully"
}
```

### 2. Train Model (`POST /train`)
Trains the logistic regression model on the encrypted dataset.

**Request Body:**
```json
{
    "email": "string",
    "password": "string"
}
```

**Response:**
```json
{
    "coef": [[float]],
    "intercept": [float],
    "scaler_mean": [float],
    "scaler_scale": [float]
}
```

### 3. Predict (`POST /predict`)
Makes predictions using the trained model.

**Request Body:**
```json
{
    "features": [float]
}
```

**Response:**
```json
{
    "prediction": int,
    "probability": [float]
}
```

## Security Features

### Homomorphic Encryption
- Uses TenSEAL CKKS scheme
- 8192 polynomial modulus degree
- Secure key generation and management
- Encrypted computations on sensitive data

### Privacy Preservation
- Data encryption at rest and in transit
- Secure multi-party computation via PySyft
- Mock data generation for sensitive information
- Secure data asset management

## Technical Requirements

### Dependencies
- Python 3.8+
- FastAPI 0.111.0
- PySyft 0.9.0
- TenSEAL 0.3.14
- scikit-learn 1.3.2
- Additional requirements as specified in requirements.txt

### Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Data Processing

### Dataset
- Uses the breast cancer dataset from scikit-learn
- Features include various biological markers
- Binary classification (benign/malignant)
- Standardized using StandardScaler

### Model
- Logistic Regression classifier
- Standard scaling of features
- Training/test split (80/20)
- Probability-based predictions

## Error Handling
- Comprehensive exception handling
- HTTP status codes for different error scenarios
- Detailed error messages for debugging
- Graceful failure handling

## Best Practices
1. Use secure credentials for authentication
2. Regularly update encryption keys
3. Monitor model performance
4. Keep dependencies updated
5. Follow security protocols for data handling

## Limitations
1. Currently supports only logistic regression
2. Limited to breast cancer dataset
3. Requires specific version compatibility
4. Performance overhead due to encryption
