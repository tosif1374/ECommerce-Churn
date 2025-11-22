import os
import sys
from dataclasses import dataclass

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging as log
from src.utils import save_object  # evaluate_models should return {model_name: score}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        """
        Expects train_array and test_array with last column as label.
        Returns: best model accuracy on test set (float)
        """
        try:
            log.info("Split training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            # parameter grids for a possible GridSearch inside evaluate_models, if you use your own evaluate_models adapt accordingly
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                },
                "K-Neighbors Classifier": {
                    "n_neighbors": [3, 5, 7]
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200]
                },
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100]
                },
                "CatBoost Classifier": {
                    "depth": [6, 8],
                    "iterations": [50, 100]
                },
                "AdaBoost Classifier": {
                    "n_estimators": [50, 100]
                }
            }

            # If you have evaluate_models utility that accepts these arguments, use it:
            try:
                model_report: dict = evaluate_models(
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    models=models, param=params
                )
                log.info(f"Model evaluation report: {model_report}")
            except Exception as eval_exc:
                # Fallback: simple evaluation without hyperparameter tuning
                log.warning(f"evaluate_models failed or not available, fallback to basic fit/eval: {eval_exc}")
                model_report = {}
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds)
                    model_report[name] = score

            # Select best model name and score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            log.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Refit best model on full training data (no hyperparams grid appropriate here; if evaluate_models already did grid-search, you may have to refit using the best estimator it returned — adjust if evaluate_models returns fitted estimator)
            log.info("Fitting best model on full training data before saving")
            best_model.fit(X_train, y_train)

            # Save the trained model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            log.info(f"Saved trained model at: {self.model_trainer_config.trained_model_file_path}")

            # Evaluate and return accuracy
            predictions = best_model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            log.info(f"Best model test accuracy: {acc}")

            return acc

        except Exception as e:
            log.error(f"Exception in model trainer: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # quick local test block (requires output of data transformation to exist)
    from src.components.data_transformation import DataTransformation
    from src.components.data_ingestion import DataIngestion

    try:
        ing = DataIngestion()
        train_path, test_path = ing.initiate_data_ingestion()
        dt = DataTransformation()
        train_arr, test_arr, _ = dt.initiate_data_transformation(train_path, test_path)

        trainer = ModelTrainer()
        score = trainer.initiate_model_trainer(train_arr, test_arr)
        print(f"Model training finished. Test accuracy: {score:.4f}")
    except Exception as err:
        print(f"ERROR: {err}")
