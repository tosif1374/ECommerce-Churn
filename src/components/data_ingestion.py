import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging as log
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self, csv_path: str = None):
        """
        csv_path: optional path to dataset CSV.
        If None → tries common project locations.
        """
        self.ingestion_config = DataIngestionConfig()
        self.csv_path = csv_path

    def _find_csv(self):
        """Locate the dataset intelligently."""
        if self.csv_path:
            candidates = [self.csv_path]
        else:
            candidates = [
                os.path.join(os.getcwd(), "notebook", "Ecommerce_data_cleaned.csv"),
                os.path.join(os.getcwd(), "data", "Ecommerce_data_cleaned.csv"),
                os.path.join(os.getcwd(), "Ecommerce_data_cleaned.csv"),
                os.path.join(os.getcwd(), "datasets", "Ecommerce_data_cleaned.csv"),
            ]

        for c in candidates:
            if c and os.path.exists(c):
                log.info(f"[FOUND CSV] {c}")
                return c

        err = "\n".join(candidates)
        raise FileNotFoundError(
            "CSV file NOT found. Searched:\n"
            f"{err}\n\n"
            "Pass path explicitly → DataIngestion(csv_path='yourfile.csv')"
        )

    def initiate_data_ingestion(self):
        """Main ingestion pipeline."""
        log.info("=== ENTERING Data Ingestion Component ===")
        try:
            csv_path = self._find_csv()
            df = pd.read_csv(csv_path)
            log.info(f"Loaded dataset → Shape: {df.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            log.info(f"Saved RAW data → {self.ingestion_config.raw_data_path}")

            # Stratify only if 'churn' exists
            stratify_col = df["Churn"] if "Churn" in df.columns else None

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=stratify_col
            )

            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            log.info(f"TRAIN data saved → {self.ingestion_config.train_data_path}")
            log.info(f"TEST data saved → {self.ingestion_config.test_data_path}")
            log.info("=== DATA INGESTION COMPLETED ===")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            log.error(f"Data Ingestion ERROR → {str(e)}")
            raise CustomException(e, sys)



if __name__ == "__main__":
    try:
        log.info("=== PIPELINE RUN STARTED ===")
        
        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        log.info("STEP 1 → Data Ingestion Done")

        # 2️⃣ Data Transformation
        transformer = DataTransformation()
        train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        log.info("STEP 2 → Data Transformation Done")

        # 3️⃣ Model Training
        trainer = ModelTrainer()
        training_result = trainer.initiate_model_trainer(train_arr, test_arr)
        log.info(f"STEP 3 → Model Training Completed → Result: {training_result}")

        log.info("=== FULL PIPELINE SUCCESS ===")

    except Exception as e:
        log.error(f"PIPELINE FAILED → {e}")
        print(f"ERROR: {e}")
