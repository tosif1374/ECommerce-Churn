import logging
import mlflow
import mlflow.sklearn

from src.pipeline.train_pipeline import run_pipeline
from src.pipeline.predict_pipeline import PredictPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def main():

    mlflow.set_experiment("Ecommerce_Churn")

    with mlflow.start_run() as run:

        logging.info("Training started")

        # Train model
        accuracy = run_pipeline()

        logging.info("Training completed")


        # Load prediction pipeline
        predict_pipeline = PredictPipeline()

        logging.info("Prediction pipeline ready")


        # ✅ REQUIRED FIX: Log metrics and model properly

        mlflow.log_param("model", "catboost")

        mlflow.log_metric("accuracy", accuracy)


        # Log saved model
        import joblib

        model = joblib.load("artifacts/model.pkl")

        mlflow.sklearn.log_model(
    model,
    "model"
     )


        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "Ecommerce_Churn_Model")


        logging.info("MLflow logging completed")


if __name__ == "__main__":

    main()