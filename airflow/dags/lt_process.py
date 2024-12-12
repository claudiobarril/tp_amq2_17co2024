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
        import numpy as np
        import logging

        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_original_path = Variable.get("cars_dataset_location")
        dataset = wr.s3.read_csv(data_original_path)

        logger = logging.getLogger("airflow.task")
        logger.info("Columnas del dataset: %s", dataset.columns)

        dataset.drop_duplicates(keep='first', inplace=True)
        dataset["selling_price_log"] = np.log1p(dataset["selling_price"])

        X = dataset.drop(columns=['selling_price', 'selling_price_log'])
        y_log = dataset['selling_price_log']

        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=42)

        X_train_path = Variable.get("cars_X_train_location")
        X_test_path = Variable.get("cars_X_test_location")

        save_to_csv(X_train, X_train_path)
        save_to_csv(X_test, X_test_path)
        save_to_csv(y_train, Variable.get("cars_y_train_location"))
        save_to_csv(y_test, Variable.get("cars_y_test_location"))

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
        import os
        import pandas as pd

        from sklearn.experimental import enable_iterative_imputer
        from feature_engineering.cars_pipeline import CarsPipeline
        from airflow.models import Variable
        from models.prediction_key import PredictionKey
        from joblib import load

        logger = logging.getLogger("airflow.task")
        logger.info("X_train_path: %s", X_train_path)
        logger.info("X_test_path: %s", X_test_path)

        X_train = wr.s3.read_csv(X_train_path)
        y_train = wr.s3.read_csv(Variable.get("cars_y_train_location"))
        X_test = wr.s3.read_csv(X_test_path)

        if pd.notna(X_test.iloc[0]).all():  # Check for missing values in first row
            logger.info("First complete record of X_train:\n%s", X_test.iloc[1])
        else:
            logger.warning("First row of X_train is incomplete.")

        final_pipeline = CarsPipeline()
        final_pipeline.fit(X_train, y_train)
        X_train_processed = final_pipeline.fit_transform_df(X_train)
        X_test_processed = final_pipeline.transform_df(X_test)

        key2, hashes2 = PredictionKey().from_pipeline(final_pipeline.transform(X_test.iloc[1:2]))
        logger.info(f'prediction hash [{hashes2[0]}] key[{key2[0]}]')

        wr.s3.to_csv(df=X_train_processed, path=Variable.get("cars_X_train_processed_location"), index=False)
        wr.s3.to_csv(df=X_test_processed, path=Variable.get("cars_X_test_processed_location"), index=False)

        try:
            X_to_predict = wr.s3.read_csv(Variable.get("cars_X_to_predict_location"))
            X_to_predict_processed = final_pipeline.transform_df(X_to_predict)
            combined_processed = pd.concat([X_train_processed, X_test_processed, X_to_predict_processed])
        except wr.exceptions.NoFilesFound as e:
            logger.info('no request to predict pending')
            combined_processed = pd.concat([X_train_processed, X_test_processed])

        wr.s3.to_csv(df=combined_processed, path=Variable.get("cars_X_combined_processed_location"), index=False)

        pipeline_path = "/tmp/final_pipeline.joblib"
        try:
            joblib.dump(final_pipeline, pipeline_path)
            logger.info("Pipeline serialized to %s", pipeline_path)

            s3 = boto3.client('s3')
            bucket_name = "data"
            object_key = Variable.get("pipeline_object_key")

            s3.upload_file(pipeline_path, bucket_name, object_key)
            logger.info("Pipeline uploaded to s3://%s/%s", bucket_name, object_key)
        except Exception as e:
            logger.error(e)
            logger.error(f"An error occurred: {e}")
        finally:
            # Clean up: delete the temporary file after uploading to S3
            if os.path.exists(pipeline_path):
                os.remove(pipeline_path)
                logger.info("Temporary file %s deleted.", pipeline_path)


    split_dataset_output = split_dataset()
    feature_engineering(split_dataset_output["X_train_path"], split_dataset_output["X_test_path"])


dag = process_lt_cars_data()
