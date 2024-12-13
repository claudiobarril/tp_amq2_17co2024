import logging
import pickle

import mlflow


class ModelLoader():

    def __init__(self, model_name, alias):
        self.model_name = model_name
        self.alias = alias
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        mlflow.set_tracking_uri('http://mlflow:5002')
        self.mlflow_client = mlflow.tracking.MlflowClient()

    def load_model(self):
        try:
            model_data_mlflow = self.mlflow_client.get_model_version_by_alias(self.model_name, self.alias)
            self.model_ml = mlflow.xgboost.load_model(model_data_mlflow.source)
            self.version_model_ml = int(model_data_mlflow.version)
            self.logger.info("Modelo cargado correctamente desde MLFlow.")
        except:
            file_ml = open('/app/models/best_xgboost_model.pkl', 'rb')
            self.model_ml = pickle.load(file_ml)
            file_ml.close()
            self.version_model_ml = 0
            self.logger.info("Modelo cargado correctamente desde disco.")

    def check_model(self):
        try:
            new_model_data = self.mlflow_client.get_model_version_by_alias(self.model_name, self.alias)
            new_version_model = int(new_model_data.version)

            if new_version_model != self.version_model_ml:
                self.load_model()

        except Exception as e:
            self.logger.error(f"error checking model version update err[{e}]")
            pass
