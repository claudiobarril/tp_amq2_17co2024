import datetime
import logging

from airflow.decorators import dag, task
from sklearn.experimental import enable_iterative_imputer
from feature_engineering.cars_pipeline import CarsPipeline

markdown_text = """
### LT Process for Cars Data
"""

default_args = {
    'owner': "17co2024",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15),
    'catchup': False,  # Evitar trabajos pendientes
}


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
        system_site_packages=True
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
        y = dataset['selling_price']
        y_log = dataset['selling_price_log']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.3, random_state=42)

        save_to_csv(X_train, "s3://data/final/train/cars_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/cars_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/cars_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/cars_y_test.csv")

    @task.virtualenv(
        task_id="feature_engineering",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def feature_engineering():
        """
        Convert categorical variables into one-hot encoding.
        """
        import pandas as pd
        import awswrangler as wr
        import logging
        from sklearn.experimental import enable_iterative_imputer
        from feature_engineering.cars_pipeline import CarsPipeline

        logger = logging.getLogger("airflow.task")

        X_train = wr.s3.read_csv("s3://data/final/train/cars_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/cars_X_test.csv")

        logger.info("Columnas del dataset X_train: %s", X_train.columns)
        logger.info("Columnas del dataset X_test: %s", X_test.columns)

        final_pipeline = CarsPipeline()
        X_train_processed = final_pipeline.fit_transform_df(X_train)
        X_test_processed = final_pipeline.transform_df(X_test)

        wr.s3.to_csv(df=X_train_processed, path="s3://data/final/train/cars_X_train_processed.csv", index=False)
        wr.s3.to_csv(df=X_test_processed, path="s3://data/final/test/cars_X_test_processed.csv", index=False)

    split_dataset() >> feature_engineering()


dag = process_lt_cars_data()
