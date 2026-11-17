import os 
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline

from fraud_detection.entity.config_entity import DataTransformationConfig
from fraud_detection.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact, DataIngestionArtifact
from fraud_detection.exception.exception import FraudDetectionException
from fraud_detection.logging.logger import logging
from fraud_detection.utils.main_utils import save_numpy_array_data, load_numpy_array_data, read_yaml_file, save_object
from fraud_detection.constants.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH
from fraud_detection.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS


class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact,
                 data_ingestion_artifact:DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self._schema_config_ = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise FraudDetectionException(e, sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            logging.info('entered get_data_transformer_object')

            numerical_columns = self._schema_config_.get("num_columns",[])
            categorical_columns = self._schema_config_.get("cat_columns",[])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            num_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps = [
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            logging.info('Created preprocessor columntransformer successfully')
            return preprocessor
        
        except Exception as e:
            raise FraudDetectionException(e, sys)


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info("entered initiate_data_transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception('Data validation failed!')

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            preprocessor_obj = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]

            drop_columns = self._schema_config_.get("drop_columns",[])
            for col in drop_columns:
                if col in input_feature_train_df:
                    input_feature_train_df = input_feature_train_df.drop(columns=[col])
                if col in input_feature_test_df:
                    input_feature_test_df = input_feature_test_df.drop(columns=[col])

            logging.info('applying preprocessing to train test input features')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)

            save_object(self.data_transformation_config.preprocessed_object_file_path, preprocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.preprocessed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info('datatransformation completed')

            return data_transformation_artifact

        except Exception as e:
            raise FraudDetectionException(e, sys)