import joblib

def load_model(path):
    """
    Load a saved model from disk.
    
    Parameters:
    - path: Path to .pkl model file
    
    Returns:
    - model: Loaded model object
    """
    return joblib.load(path)

def predict(model, X):
    """
    Predict churn labels using a trained model.
    
    Parameters:
    - model: Trained model
    - X: DataFrame of input features
    
    Returns:
    - Predictions (array)
    """
    return model.predict(X)
