import mlflow

# Clases utilitarias para la gestión de modelos y experimentos
class ModelManager:
    def __init__(self, model_name="cars_model_prod"):
        self.model_name = model_name
        self.client = mlflow.MlflowClient()

    def load_model(self, alias):
        model_data = self.client.get_model_version_by_alias(self.model_name, alias)
        model = mlflow.xgboost.load_model(model_data.source)
        return model

    def register_challenger(self, model, neg_mse, model_uri):
        # Save the model params as tags
        tags = model.get_params()
        tags["model"] = type(model).__name__
        tags["neg_MSE"] = neg_mse

        # Save the version of the model
        result = self.client.create_model_version(
            name=self.model_name,
            source=model_uri,
            run_id=model_uri.split("/")[-3],
            tags=tags
        )

        # Save the alias as challenger
        self.client.set_registered_model_alias(self.model_name, "challenger", result.version)

    def promote_challenger(self):
        # Demote the champion
        self.client.delete_registered_model_alias(self.model_name, "champion")

        # Load the challenger from registry
        challenger_version = self.client.get_model_version_by_alias(self.model_name, "challenger")

        # delete the alias of challenger
        self.client.delete_registered_model_alias(self.model_name, "challenger")

        # Transform in champion
        self.client.set_registered_model_alias(self.model_name, "champion", challenger_version.version)

    def demote_challenger(self):
        self.client.delete_registered_model_alias(self.model_name, "challenger")
