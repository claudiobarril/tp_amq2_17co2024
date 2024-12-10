import datetime
import mlflow
from mlflow.models import infer_signature

class ExperimentTracker:
    def __init__(self, experiment_name="Cars"):
        self.experiment_name = experiment_name

    def track_experiment(self, model, model_name, X):
        # Track the experiment
        experiment = mlflow.set_experiment(self.experiment_name)

        mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "challenger models", "dataset": "Cars"},
                         log_system_metrics=True)

        params = model.get_params()
        params["model"] = type(model).__name__

        mlflow.log_params(params)

        # Save the artifact of the challenger model
        artifact_path = "model"

        signature = infer_signature(X, model.predict(X))

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path=artifact_path,
            signature=signature,
            registered_model_name=model_name,
            metadata={"model_data_version": 1}
        )

        # Obtain the model URI
        return mlflow.get_artifact_uri(artifact_path)
