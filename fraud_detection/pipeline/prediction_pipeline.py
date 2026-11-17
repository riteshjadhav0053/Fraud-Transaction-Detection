import os 
import sys 
import pandas as pd

from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.utils.main_utils import load_object
from fraud_detection.entity.config_entity import TrainingPipelineConfig, DataTransformationConfig

class PredictionPipeline:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def get_latest_model_path(self) -> tuple:
        try:
            artifacts_dir = "Artifacts"
            if not os.path.exists(artifacts_dir):
                raise Exception("Artifacts directory does not exist. Run training pipeline first!")

            timestamp_dirs = [os.path.join(artifacts_dir, d) for d in os.listdir(artifacts_dir)
                              if os.path.isdir(os.path.join(artifacts_dir, d))]

            if not timestamp_dirs:
                raise Exception("No timestamp folder found.")

            latest_timestamp_dir = max(timestamp_dirs, key=os.path.getctime)

            # Search dynamically for preprocessor object anywhere in the timestamp folder
            preprocessor_path = None
            for root, dirs, files in os.walk(latest_timestamp_dir):
                if "preprocessor.pkl" in files or "preprocessing.pkl" in files:
                    preprocessor_path = os.path.join(root, files[files.index("preprocessor.pkl") if "preprocessor.pkl" in files else files.index("preprocessing.pkl")])
                    break
            
            # Fallback explicit check based on config structure
            if not preprocessor_path or not os.path.exists(preprocessor_path):
                preprocessor_path = os.path.join(latest_timestamp_dir, "data_transformation", "transformed_object", "preprocessing.pkl")

            if not os.path.exists(preprocessor_path):
                preprocessor_path = os.path.join(latest_timestamp_dir, "data_transformation", "preprocessing", "preprocessor.pkl")

            model_path = os.path.join(latest_timestamp_dir, "model_trainer", "trained_model", "model.pkl")

            logging.info(f"Resolved preprocessor path: {preprocessor_path}")
            logging.info(f"Resolved model path: {model_path}")
            
            return preprocessor_path, model_path
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def predict(self, dataframe: pd.DataFrame):
        try:
            logging.info("Executing prediction on incoming data...")
            
            # Fetch paths dynamically using helper method
            preprocessor_path, model_path = self.get_latest_model_path()
            
            # Load objects using correct string paths
            logging.info(f"Loading preprocessor from: {preprocessor_path}")
            preprocessor = load_object(preprocessor_path)
            
            logging.info(f"Loading model from: {model_path}")
            model = load_object(model_path)
            
            # Transform raw input dataframe
            transformed_feature = preprocessor.transform(dataframe)
            
            # Make predictions
            predictions = model.predict(transformed_feature)
            probabilities = model.predict_proba(transformed_feature) if hasattr(model, "predict_proba") else None
            
            return predictions, probabilities
        except Exception as e:
            raise FraudDetectionException(e, sys)


class CustomData:
    def __init__(self,
                 step: int,
                 type: str,
                 amount: float,
                 oldbalanceOrg: float,
                 newbalanceOrig: float,
                 oldbalanceDest: float,
                 newbalanceDest: float):
        self.step = step
        self.type = type
        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig
        self.oldbalanceDest = oldbalanceDest
        self.newbalanceDest = newbalanceDest

    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            input_data_dict = {
                "step": [self.step],
                "type": [self.type],
                "amount": [self.amount],
                "oldbalanceOrg": [self.oldbalanceOrg],
                "newbalanceOrig": [self.newbalanceOrig],
                "oldbalanceDest": [self.oldbalanceDest],
                "newbalanceDest": [self.newbalanceDest],
            }
            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise FraudDetectionException(e, sys)