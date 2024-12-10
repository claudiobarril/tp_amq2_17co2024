from airflow.decorators import dag, task
from config.default_args import default_args

markdown_text = """
### LT Process for Cars Data
"""


@dag(
    dag_id="process_lt_cars_data",
    description="LT process for cars data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["LT", "Cars"],
    default_args=default_args,
    catchup=False,
)
def process_lt_cars_data():
    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0", "scikit-learn==1.3.2", "numpy==1.26.4"],
        system_site_packages=True,
        multiple_outputs=True,
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part.
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        import numpy as np
        import logging

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_original_path = "s3://data/raw/cars.csv"
        dataset = wr.s3.read_csv(data_original_path)

        logger = logging.getLogger("airflow.task")
        logger.info("Columnas del dataset: %s", dataset.columns)

        dataset.drop_duplicates(keep='first', inplace=True)
        dataset["selling_price_log"] = np.log1p(dataset["selling_price"])

        X = dataset.drop(columns=['selling_price', 'selling_price_log'])
        y_log = dataset['selling_price_log']

        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)

        X_train_path = "s3://data/final/train/cars_X_train.csv"
        X_test_path = "s3://data/final/test/cars_X_test.csv"

        save_to_csv(X_train, X_train_path)
        save_to_csv(X_test, X_test_path)
        save_to_csv(y_train, "s3://data/final/train/cars_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/cars_y_test.csv")

        return {"X_train_path": X_train_path, "X_test_path": X_test_path}

    @task.virtualenv(
        task_id="feature_engineering",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def feature_engineering(X_train_path, X_test_path):
        """
        Convert categorical variables into one-hot encoding.
        """
        import awswrangler as wr
        import logging
        import boto3
        import joblib

        from sklearn.experimental import enable_iterative_imputer
        from feature_engineering.cars_pipeline import CarsPipeline

        logger = logging.getLogger("airflow.task")
        logger.info("X_train_path: %s", X_train_path)
        logger.info("X_test_path: %s", X_test_path)

        X_train = wr.s3.read_csv(X_train_path)
        X_test = wr.s3.read_csv(X_test_path)

        logger.info("[ANTES] Columnas del dataset X_train: %s", X_train.columns)
        logger.info("[ANTES] Columnas del dataset X_test: %s", X_test.columns)

        # Log the first few rows of X_train and X_test
        logger.info("Primeras filas de X_train:\n%s", X_train.head().to_string(index=False))
        logger.info("Primeras filas de X_test:\n%s", X_test.head().to_string(index=False))

        final_pipeline = CarsPipeline()
        X_train_processed = final_pipeline.fit_transform_df(X_train)
        X_test_processed = final_pipeline.transform_df(X_test)

        logger.info("[DESPUES] Columnas del dataset X_train: %s", X_train_processed.columns)
        logger.info("[DESPUES] Columnas del dataset X_test: %s", X_test_processed.columns)

        wr.s3.to_csv(df=X_train_processed, path="s3://data/final/train/cars_X_train_processed.csv", index=False)
        wr.s3.to_csv(df=X_test_processed, path="s3://data/final/test/cars_X_test_processed.csv", index=False)

        # Serialize the final_pipeline using joblib
        pipeline_path = "/tmp/final_pipeline.joblib"
        joblib.dump(final_pipeline, pipeline_path)

        # Save the serialized pipeline to S3
        s3 = boto3.client('s3')
        bucket_name = "data"
        object_key = "final/pipeline/final_pipeline.joblib"

        s3.upload_file(pipeline_path, bucket_name, object_key)

    split_dataset_output = split_dataset()
    feature_engineering(split_dataset_output["X_train_path"], split_dataset_output["X_test_path"])


dag = process_lt_cars_data()
