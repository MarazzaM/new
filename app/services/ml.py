from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def train_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return {
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }

def prepare_model(model_data):
    model = LogisticRegression()
    model.coef_ = np.array(model_data["coef"])
    model.intercept_ = np.array(model_data["intercept"])
    model.classes_ = np.array([0, 1])
    
    scaler = StandardScaler()
    scaler.mean_ = np.array(model_data["scaler_mean"])
    scaler.scale_ = np.array(model_data["scaler_scale"])
    
    return model, scaler