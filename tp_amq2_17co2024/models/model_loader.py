import mlflow
import pickle

class ModelLoader():

    def __init__(self, model_name, alias):
        self.model_name = model_name
        self.alias = alias

    def load_model(self):
        try:
            mlflow.set_tracking_uri('http://mlflow:5001')
            # Load the trained model from MLflow
            client_mlflow = mlflow.MlflowClient()

            model_data_mlflow = client_mlflow.get_model_version_by_alias(self.model_name, self.alias)
            self.model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
            self.version_model_ml = int(model_data_mlflow.version)
        except:
            # If there is no registry in MLflow, open the default model
            file_ml = open('/app/files/best_xgboost_model.pkl', 'rb')
            self.model_ml = pickle.load(file_ml)
            file_ml.close()
            self.version_model_ml = 0

    def check_model(self):
        """
        Check for updates in the model and update if necessary.

        The function checks the model registry to see if the version of the champion model has changed. If the version
        has changed, it updates the model and the data dictionary accordingly.

        :return: None
        """
        try:
            mlflow.set_tracking_uri('http://mlflow:5001')
            client = mlflow.MlflowClient()

            # Check in the model registry if the version of the champion has changed
            new_model_data = client.get_model_version_by_alias(self.model_name, self.alias)
            new_version_model = int(new_model_data.version)

            # If the versions are not the same
            if new_version_model != self.version_model:
                # Load the new model and update version and data dictionary
                self.load_model()

        except:
            # If an error occurs during the process, pass silently
            pass