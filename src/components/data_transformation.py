import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging as log
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_cols = [
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
            ]

            categorical_cols = [
                'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus'
            ]

            log.info(f"Categorical columns: {categorical_cols}")
            log.info(f"Numerical columns: {numerical_cols}")

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numeric_pipeline, numerical_cols),
                    ("cat_pipeline", categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor, numerical_cols, categorical_cols

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            log.info("Read train and test data completed")

            target_column = "Churn"

            preprocessor, num_cols, cat_cols = self.get_data_transformer()

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            log.info("Fitting and transforming training features")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            log.error(f"Exception in data transformation: {e}")
            raise CustomException(e, sys)
