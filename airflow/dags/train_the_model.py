from airflow.decorators import dag, task
from config.default_args import default_args

markdown_text = """
### Train the Model for Cars Data

This DAG trains the model based on existing data, and put in production.
"""

@dag(
    dag_id="train_the_model",
    description="Train the model based on existing data, and put in production",
    doc_md=markdown_text,
    tags=["Train", "Cars"],
    default_args=default_args,
    catchup=False,
)
def train_the_model():

    @task
    def get_or_create_experiment(experiment_name):
        """Get or create an experiment in MLflow."""
        import mlflow

        from airflow.models import Variable

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        return mlflow.create_experiment(experiment_name)

    @task
    def train_and_log_model(experiment_id):
        """Train and log the model with MLflow."""
        import datetime
        import numpy as np
        import mlflow
        import logging

        from mlflow import MlflowClient
        from sklearn.model_selection import RandomizedSearchCV, KFold
        from xgboost import XGBRegressor
        from sklearn.metrics import (
            mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
        )
        from mlflow.models.signature import infer_signature
        from scipy.stats import uniform, randint
        from airflow.models import Variable
        from models.data_loader import load_train_test_data

        logger = logging.getLogger("airflow.task")
        logger.info("Experiment id: %s", experiment_id)

        # Load data
        X_train, y_train, X_test, y_test = load_train_test_data()

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        run_name_parent = "best_hyperparam_" + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')

        model = XGBRegressor(random_state=42)
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'gamma': uniform(0, 0.5),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(1, 2)
        }

        xgb = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=50,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            random_state=42,
        )

        xgb.fit(X_train, y_train)
        xgb_best_model = xgb.best_estimator_

        # Predictions and metrics
        y_pred = np.expm1(xgb_best_model.predict(X_test))
        y_pred_train = np.expm1(xgb_best_model.predict(X_train))
        y_train_recovered = np.expm1(y_train.to_numpy().flatten())
        y_test_recovered = np.expm1(y_test.to_numpy().flatten())

        metrics = {
            "MAE_training": mean_absolute_error(y_train_recovered, y_pred_train),
            "MAE": mean_absolute_error(y_test_recovered, y_pred),
            "RMSE": root_mean_squared_error(y_test_recovered, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test_recovered, y_pred),
            "R2": r2_score(y_test_recovered, y_pred)
        }

        logger.info("Metrics: %s", metrics)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent, nested=True):
            mlflow.log_params(xgb.best_params_)
            mlflow.log_metrics(metrics)
            artifact_path = "model"
            signature = infer_signature(X_train, xgb_best_model.predict(X_train))

            mlflow.xgboost.log_model(
                xgb_model=xgb_best_model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name="cars_model_dev",
                input_example=X_train.head(),
                metadata={"model_data_version": 1},
                extra_pip_requirements=["xgboost==2.1.3"]
            )

            model_uri = mlflow.get_artifact_uri(artifact_path)
            logger.info("Model uri: %s", model_uri)

            client = MlflowClient()
            name = "cars_model_prod"
            desc = "This model predicts selling price for used cars"

            # Check if the model already exists, if not, create it
            try:
                model_registered = client.get_registered_model(name)
            except mlflow.exceptions.MlflowException as e:
                if "RESOURCE_NOT_FOUND" in str(e):
                    model_registered = None
                else:
                    raise e

            if not model_registered:
                # If the model doesn't exist, create it
                client.create_registered_model(name=name, description=desc)

            # Guardamos como tag los hiper-parametros en la version del modelo
            tags = xgb_best_model.get_params()
            tags["model"] = type(xgb_best_model).__name__
            tags["mae_training"] = metrics["MAE_training"]
            tags["mae"] = metrics["MAE"]
            tags["rmse"] = metrics["RMSE"]
            tags["mape"] = metrics["MAPE"]
            tags["r2"] = metrics["R2"]

            # Guardamos la version del modelo
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Create alias for champion version
            client.set_registered_model_alias(name, "champion", result.version)

    # Workflow
    experiment_id = get_or_create_experiment("Cars")
    train_and_log_model(experiment_id)

# Register the DAG
train_the_model()
