import sys
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        logger.info("=== PIPELINE RUN STARTED ===")

        ingestion_obj = DataIngestion()
        train_path, test_path = ingestion_obj.initiate_data_ingestion()
        logger.info(" Data Ingestion Done")

        transformation_obj = DataTransformation()
        train_arr, test_arr, preprocessor = transformation_obj.initiate_data_transformation(
    train_path, test_path
)

        logger.info(" Data Transformation Done")

        trainer_obj = ModelTrainer()
        accuracy = trainer_obj.initiate_model_trainer(train_arr, test_arr)
        logger.info(f"Model Training Completed → Result: {accuracy}")

        logger.info("=== FULL PIPELINE SUCCESS ===")

    except Exception as e:
        logger.error("PIPELINE FAILED")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_pipeline()
