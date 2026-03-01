import sys
import pandas as pd
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

        # ✅ REQUIRED FIX ONLY
        train_path = "artifacts/train_enriched.csv"

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
    return accuracy

from src.scraping.snapdeal_scraper import scrape_snapdeal
from src.utils import clean_scraped_data


# 1️ Scrape external data
scrape_snapdeal()

# 2️Clean & generate features
logger.info("Cleaning scraped data...")
scraped_df = clean_scraped_data()

# 3️ Load your existing training data
train_df = pd.read_csv("artifacts/train.csv")

# 4️ Merge logic (simple global feature)
train_df["high_price"] = scraped_df["high_price"].iloc[0]

# 5️ Save updated training data
train_df.to_csv("artifacts/train_enriched.csv", index=False)
logger.info("Scraped data cleaned and merged into training data.")

if __name__ == "__main__":
    run_pipeline()