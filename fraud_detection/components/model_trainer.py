import os 
import sys 
import pandas as pd 
import numpy as np

from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from fraud_detection.entity.config_entity import ModelTrainerConfig
from fraud_detection.utils.main_utils import save_object, load_object

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


class ModelTrainer:
    def __init__(self, data_transformation_artifact : DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config

        except Exception as e:
            raise FraudDetectionException(e, sys)

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("entered initiate_model_trainer")

            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info("training model with randomforestclassifier")
            model = RandomForestClassifier(random_state=42, max_depth=10)  
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)    


            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            logging.info(f"Train F1 Score: {train_f1}")
            logging.info(f"Test F1 Score: {test_f1}")   

            if test_f1 < self.model_trainer_config.expected_score:
                raise Exception(f"Model score [{test_f1}] is below expected score [{self.model_trainer_config.expected_score}].")

            # Check for overfitting
            if abs(train_f1 - test_f1) > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(f"Model may be overfitting! Difference between train and test F1 score is {abs(train_f1 - test_f1)}")

            save_object(self.model_trainer_config.trained_model_file_path, model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_f1,
                test_metric_artifact=test_f1
            )

            logging.info(f"ModelTrainerArtifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys)