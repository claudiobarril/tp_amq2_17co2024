from airflow.decorators import dag, task
from config.default_args import default_args


markdown_text = """
### Batch processing with best model

DAG for batch processing predictions
"""

@dag(
    dag_id="batch_processing_model",
    description="DAG for batch processing predictions",
    doc_md=markdown_text,
    tags=["Batch-processing", "Cars"],
    default_args=default_args,
    catchup=False,
)
def batch_processing_model():
    """DAG for batch processing predictions using best model available."""

    @task
    def batch_processing():
        from airflow.models import Variable
        import mlflow.xgboost
        import numpy as np
        import awswrangler as wr

        # Set MLflow tracking URI
        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        """Process batch data and generate predictions."""
        # Load batch data
        X_batch = wr.s3.read_csv(Variable.get("cars_X_combined_processed_location"))

        # Load model
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias("cars_model_prod", "champion")
        model = mlflow.xgboost.load_model(model_data_mlflow.source)

        # Generate predictions
        out = model.predict(X_batch)
        labels = np.array([np.exp(o) for o in out]).astype(str)

        # Create Redis-compatible data
        from models.prediction_key import PredictionKey
        keys, hashes = PredictionKey().from_dataframe(X_batch)
        X_batch['key'] = keys
        X_batch['hashed'] = hashes

        dict_redis = {
            row['hashed']: labels[idx]
            for idx, row in X_batch.iterrows()
        }

        return dict_redis

    @task
    def ingest_redis(redis_data):
        import redis
        from airflow.models import Variable

        redis_host = Variable.get("redis_host")
        redis_port = Variable.get("redis_port")

        """Ingest predictions into Redis."""
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        pipeline = r.pipeline()

        for key, value in redis_data.items():
            pipeline.set(key, value)

        pipeline.execute()

    @task
    def start_task():
        """Start task message."""
        import logging

        logger = logging.getLogger("airflow.task")
        logger.info("Starting task message")

    @task
    def end_task():
        """End task message."""
        import logging

        logger = logging.getLogger("airflow.task")
        logger.info("Finished Batch Prediction")

    # Workflow
    redis_data = start_task() >> batch_processing()
    ingest_redis(redis_data) >> end_task()

# Register the DAG
batch_processing_model()
