from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import numpy as np
import redis
from catboost import CatBoostRegressor
from models.prediction_key import PredictionKey
import boto3
import awswrangler as wr
from airflow.models import Variable

REDIS_HOST = 'redis'
REDIS_PORT = 6379


def batch_processing(**kwargs):
    s3_bucket = 'data'
    s3_key = 'artifact/best_catboost_model.json'
    local_path = '/tmp/best_catboost_model.json'

    X_batch = wr.s3.read_csv(Variable.get("cars_X_combined_processed_location"))


    s3_client = boto3.client('s3',
                             aws_access_key_id='minio',
                             aws_secret_access_key='minio123',
                             endpoint_url='http://s3:9000')

    s3_client.download_file(s3_bucket, s3_key, local_path)

    model = CatBoostRegressor()
    model.load_model(local_path)

    out = model.predict(X_batch)
    labels = np.array([np.exp(o) for o in out]).astype(str)

    keys, hashes = PredictionKey().from_dataframe(X_batch)
    X_batch['key'] = keys
    X_batch['hash'] = hashes

    dict_redis = {}
    for idx, row in X_batch.iterrows():
        dict_redis[row['hash']] = labels[idx]

    ti = kwargs['ti']
    ti.xcom_push(key='redis_data', value=dict_redis)


def ingest_redis(**kwargs):
    ti = kwargs['ti']
    redis_data = ti.xcom_pull(key='redis_data')

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    pipeline = r.pipeline()

    for key, value in redis_data.items():
        pipeline.set(key, value)

    pipeline.execute()


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'deprecated_batch_processing_model',
    default_args=default_args,
    description='DAG para procesamiento por lotes de predicciones',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

start = PythonOperator(
    task_id='start',
    python_callable=lambda: print("Starting Batch Prediction"),
    dag=dag
)

batch_processing_task = PythonOperator(
    task_id='batch_processing',
    python_callable=batch_processing,
    provide_context=True,
    dag=dag
)

ingest_redis_task = PythonOperator(
    task_id='ingest_redis',
    python_callable=ingest_redis,
    provide_context=True,
    dag=dag
)

end = PythonOperator(
    task_id='end',
    python_callable=lambda: print("Finished"),
    dag=dag
)

start >> batch_processing_task >> ingest_redis_task >> end
