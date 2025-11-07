from src.train_model import train_xgb_model

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Predict and print evaluation metrics.
    
    Parameters:
    - model: Trained classifier
    - X_test: Test features
    - y_test: True test labels
    
    Returns:
    - y_pred: Predicted labels
    """
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred
