from fraud_detection.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        print("Training Pipeline executed successfully!")
    except Exception as e:
        print(f"Pipeline failed: {e}")