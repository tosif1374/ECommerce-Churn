import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

import pandas as pd

def clean_scraped_data(path="artifacts/snapdeal_products.csv"):
    df = pd.read_csv(path)

    df["price"] = (
        df["price"]
        .str.replace("Rs.", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # Feature from scraped data
    df["high_price"] = (df["price"] > df["price"].mean()).astype(int)

    df.to_csv("artifacts/snapdeal_features.csv", index=False)
    return df

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)