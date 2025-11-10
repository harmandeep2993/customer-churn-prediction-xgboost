import joblib
import pandas as pd
from src.preprocess import full_preprocess_pipeline


def load_model(model_path='models/xgb_churn_full_tuned.pkl'):
    """Load trained churn prediction model."""
    return joblib.load(model_path)


def predict_churn(input_df, model):
    """
    Preprocess input data and predict churn probability.

    Parameters:
        input_df (pd.DataFrame): Raw input features.
        model: Loaded trained model.

    Returns:
        pred (np.ndarray): Predicted churn class (0 or 1).
        prob (np.ndarray): Churn probability for class 1.
    """
    # Preprocess input
    processed = full_preprocess_pipeline(input_df)

    # Align columns with training data
    train_columns = joblib.load('models/train_columns.pkl')
    for col in train_columns:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[train_columns]

    # Predict
    pred = model.predict(processed)
    prob = model.predict_proba(processed)[:, 1]

    return pred, prob