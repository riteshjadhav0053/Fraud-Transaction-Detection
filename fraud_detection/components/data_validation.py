import os
import sys
import pandas as pd
import numpy as np
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
from fraud_detection.utils.main_utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e :
            raise FraudDetectionException(e, sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudDetectionException(e,sys)

    def validate_number_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config.get("columns", []))
            logging.info(f'Required number of columns: {number_of_columns}')
            logging.info(f'Data frame has: {len(dataframe.columns)} columns')
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e :
            raise FraudDetectionException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()
                
                # Only run KS test on numeric columns
                if pd.api.types.is_numeric_dtype(d1) and pd.api.types.is_numeric_dtype(d2):
                    sample_1 = d1.to_numpy()
                    sample_2 = d2.to_numpy()
                    
                    if len(sample_1) > 0 and len(sample_2) > 0:
                        result = ks_2samp(sample_1, sample_2)
                        is_found = bool(result.pvalue < threshold)
                        report[column] = {
                            "p_value": float(result.pvalue),
                            "drift_status": is_found
                        }
                        if is_found:
                            logging.info(f"Drift detected in column: {column} with p-value: {result.pvalue}")
                            #status = False
                else:
                    # Optional: Log or note non-numeric columns skipped for drift
                    report[column] = {
                        "p_value": None,
                        "drift_status": False,
                        "message": "Skipped drift check for non-numeric column"
                    }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path, report)
            return status
        except Exception as e:
            raise FraudDetectionException(e, sys)
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()
                is_same_dist=ks_2samp(d1,d2)
                
                # Ensure they are numeric before running KS test
                if d1.dtype in ['int64', 'float64'] and d2.dtype in ['int64', 'float64']:
                    # Convert explicitly to 1D numpy array to prevent scipy type errors
                    sample_1 = d1.to_numpy()
                    sample_2 = d2.to_numpy()
                    
                    if len(sample_1) > 0 and len(sample_2) > 0:
                        result = ks_2samp(sample_1, sample_2)
                        is_found = bool(result.pvalue < threshold)
                        report[column] = {
                            "p_value": float(result.pvalue),
                            "drift_status": is_found
                        }
                        if is_found:
                            status = False

                    report.update({column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                    }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path, report)
            return status
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns\n"
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns\n"\

            status = self.detect_dataset_drift(train_dataframe, test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header = True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)