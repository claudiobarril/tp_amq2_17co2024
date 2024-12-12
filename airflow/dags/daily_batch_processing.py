from airflow.decorators import dag, task
from config.default_args import default_args
from datetime import datetime

markdown_text = """
### Batch processing for daily predictions

DAG that retrieves the `cars_X_to_predict` dataset for the previous day, applies transformations using the pipeline, and processes batch predictions.
"""

@dag(
    dag_id="daily_batch_processing",
    description="DAG for daily predictions with cars dataset",
    doc_md=markdown_text,
    tags=["Batch-processing", "Cars", "Daily"],
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 1 * * *",  # Runs daily at 1:00 UTC
    catchup=False,
)
def daily_batch_processing():
    @task
    def generate_predictions():
        import mlflow.catboost
        import numpy as np
        import awswrangler as wr
        import logging

        from airflow.models import Variable
        from datetime import date, timedelta

        logger = logging.getLogger("airflow.task")

        # Determine yesterday's date
        yesterday = date.today() - timedelta(days=1)
        file_path = Variable.get("cars_X_to_predict_daily_location").format(yesterday=yesterday)

        try:
            X_to_predict = wr.s3.read_csv(file_path)
            logger.info(f"Loaded daily data from: {file_path}")
        except wr.exceptions.NoFilesFound:
            logger.warning(f"No data found for: {file_path}")
            return {"keys": [], "labels": []}  # No data found, return empty lists

        # Load model
        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        client_mlflow = mlflow.MlflowClient()
        model_data_mlflow = client_mlflow.get_model_version_by_alias("cars_catboost_model_prod", "champion")
        model = mlflow.catboost.load_model(model_data_mlflow.source)

        # Generate predictions
        predictions = model.predict(X_to_predict)
        labels = np.array([np.exp(p) for p in predictions]).astype(str)
        return {"keys": X_to_predict["key"].tolist(), "labels": labels.tolist()}

    @task
    def ingest_predictions(predictions):
        import redis
        import logging

        from airflow.models import Variable

        redis_host = Variable.get("redis_host")
        redis_port = Variable.get("redis_port")

        logger = logging.getLogger("airflow.task")

        # Connect to Redis
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        pipeline = r.pipeline()

        for key, label in zip(predictions["keys"], predictions["labels"]):
            pipeline.set(key, label)

        pipeline.execute()
        logger.info("Predictions ingested into Redis.")

    # Workflow
    predictions = generate_predictions()
    ingest_predictions(predictions)

# Register the DAG
daily_batch_processing()
