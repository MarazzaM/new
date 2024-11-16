from sklearn.datasets import load_breast_cancer
import pandas as pd

def load_breast_cancer_data():
    """Load and prepare the breast cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y