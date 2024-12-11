import hashlib
import os
import pickle

from catboost import CatBoostRegressor
from metaflow import FlowSpec, step, S3
from xgboost import XGBRegressor

from models.prediction_key import PredictionKey

# Configuración de las credenciales de acceso a AWS S3 (minio)
os.environ['AWS_ACCESS_KEY_ID'] = "minio"
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio123"
os.environ['AWS_ENDPOINT_URL_S3'] = "http://localhost:9000"

#export AWS_ACCESS_KEY_ID=minio
#export AWS_SECRET_ACCESS_KEY=minio123
#export AWS_ENDPOINT_URL_S3=http://localhost:9000


class BatchProcessingModel(FlowSpec):

    @step
    def start(self):
        """
        Step para iniciar el flujo. Imprime un mensaje de inicio y avanza.
        """
        print("Starting Batch Prediction")
        self.next(self.load_data, self.load_model)

    @step
    def load_data(self):
        """
        Paso para cargar los datos de entrada de S3
        """
        import pandas as pd

        # Se utiliza el objeto S3 para acceder a los datos desde el bucket en S3.
        s3 = S3(s3root="s3://data/")
        data_obj = s3.get("final/cars_X_test_processed.csv")
        self.X_batch = pd.read_csv(data_obj.path)
        self.next(self.batch_processing)


    # def load_model(self):
    #     try:
    #         mlflow.set_tracking_uri('http://mlflow:5002')
    #         # Load the trained model from MLflow
    #         client_mlflow = mlflow.MlflowClient()
    #
    #         model_data_mlflow = client_mlflow.get_model_version_by_alias(self.model_name, self.alias)
    #         self.model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
    #         self.version_model_ml = int(model_data_mlflow.version)
    #     except:
    #         # If there is no registry in MLflow, open the default model
    #         file_ml = open('../models/best_xgboost_model.pkl', 'rb')
    #         self.model_ml = pickle.load(file_ml)
    #         file_ml.close()
    #         self.version_model_ml = 0

    @step
    def load_model(self):
        """
        Paso para cargar el modelo previamente entrenado.
        """

        # Se utiliza el objeto S3 para acceder al modelo desde el bucket en S3.
        s3 = S3(s3root="s3://data/")
        model_param = s3.get("artifact/best_catboost_model.json")

        loaded_model = CatBoostRegressor()
        loaded_model.load_model(model_param.path)

        self.model = loaded_model
        self.next(self.batch_processing)

    @step
    def batch_processing(self, previous_tasks):
        """
        Paso para realizar el procesamiento por lotes y obtener predicciones.
        """
        import numpy as np
        import hashlib

        print("Obtaining predictions")

        data, model = (None, None)
        for task in previous_tasks:
            if hasattr(task, 'X_batch'):
                data = task.X_batch
            if hasattr(task, 'model'):
                model = task.model

        out = model.predict(data)
        labels = np.array([np.exp(o) for o in out]).astype(str)

        keys, hashes = PredictionKey().from_dataframe(data)
        data['key'] = keys
        data['hashed'] = hashes

        dict_redis = {}
        for index, row in data.iterrows():
            if row["key"].startswith("-2.3037035605721923 0.6695778652331075"):
                print(f'hash: [{row["hashed"]}] key[{row["key"]}]')

            dict_redis[row["hashed"]] = labels[index]

        self.redis_data = dict_redis

        self.next(self.ingest_redis)

    @step
    def ingest_redis(self):
        """
        Paso para ingestar los resultados en Redis.
        """
        import redis

        print("Ingesting Redis")
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # Comenzamos un pipeline de Redis
        pipeline = r.pipeline()

        # Se pre-ingresan los datos en Redis.
        for key, value in self.redis_data.items():
            pipeline.set(key, value)

        # Ahora ingestamos todos de una y dejamos que Redis resuelva de la forma más eficiente
        pipeline.execute()

        self.next(self.end)

    @step
    def end(self):
        """
        Paso final del flujo. Imprime un mensaje de finalización.
        """
        print("Finished")


if __name__ == "__main__":
    BatchProcessingModel()
