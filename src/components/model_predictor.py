import joblib
import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.components.data_preprocessing import DataPreprocessing


class ModelPredictor:
    """Handles loading model and making churn predictions."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)
        self.model_path = self.config["model"]["path"]
        self.columns_path = self.config["model"]["columns"]
        self.encoder_path = self.config["model"]["encoder"]

    def load_model(self):
        """Load trained churn prediction model."""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            model = joblib.load(self.model_path)
            return model
        except FileNotFoundError:
            self.logger.error(f"Model not found at {self.model_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def predict_churn(self, input_df: pd.DataFrame, model):
        """
        Preprocess input data and predict churn probability.

        Parameters:
            input_df (pd.DataFrame): Raw input features.
            model: Loaded trained model.

        Returns:
            tuple: (pred, prob)
        """
        try:
            self.logger.info("Starting churn prediction.")

            # Load encoder and preprocess
            encoder = joblib.load(self.encoder_path)
            preprocessor = DataPreprocessing()
            processed = preprocessor.full_preprocess_pipeline(input_df, encoder=encoder, fit_encoder=False)

            # Align with training columns
            train_columns = joblib.load(self.columns_path)
            for col in train_columns:
                if col not in processed.columns:
                    processed[col] = 0
            processed = processed[train_columns]

            # Predict
            pred = model.predict(processed)
            prob = model.predict_proba(processed)[:, 1]

            self.logger.info("Prediction complete.")
            return pred, prob

        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise


if __name__ == "__main__":
    predictor = ModelPredictor()
    model = predictor.load_model()

    sample_input = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 400.0
    }])

    pred, prob = predictor.predict_churn(sample_input, model)
    print(f"Prediction: {'Churn' if pred[0] else 'No Churn'} | Probability: {prob[0]:.2f}")