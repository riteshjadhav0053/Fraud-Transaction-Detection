from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def export_local_csv_as_dataframe(self):
        try:
            logging.info('exporting data into dataframe')
            df = pd.read_csv(self.data_ingestion_config.source_file_path)

            if 'id' in df.columns.tolist():
                df.drop(columns=['id'], axis=1)
            df.replace(to_replace="na", value=np.nan,inplace=True)
            return df

        except Exception as e:
            raise FraudDetectionException(e, sys)

    def export_data_into_feature_store(self, df: pd.DataFrame):
        try:
            logging.info('exporting data into feature store')
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)

            df.to_csv(feature_store_file_path, index=False, header=True)

            return df
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    def split_data_as_train_test(self, df:pd.DataFrame):
        try:
            logging.info('splitting data into train and test')
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )

            dir_name = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_name, exist_ok=True)

            logging.info('splitting data into train and test started')
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index = False, header=True
            )
            logging.info('splitting data into train and test completed')
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def initiate_data_ingestion(self):
        try:
            df = self.export_local_csv_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)
            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
            )

            return dataingestionartifact
        except Exception as e:
            raise FraudDetectionException(e, sys)
        
    