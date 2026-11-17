import sys
from fraud_detection.logging.logger import logging
from fraud_detection.exception.exception import FraudDetectionException

from fraud_detection.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from fraud_detection.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.data_transformation import DataTransformation
from fraud_detection.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        try:
            self.training_pipeline_config = TrainingPipelineConfig()
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion from training pipeline")
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation from training pipeline")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_data_transformation(self, 
                                  data_ingestion_artifact: DataIngestionArtifact, 
                                  data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation from training pipeline")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer from training pipeline")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=model_trainer_config
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed. Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Training pipeline execution started")
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logging.info(f"Training pipeline execution completed successfully. Final Model F1-Score: {model_trainer_artifact.test_metric_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise FraudDetectionException(e, sys)