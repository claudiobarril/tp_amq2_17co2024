from airflow.decorators import dag, task
from config.default_args import default_args

markdown_text = """
### Train the Model for Cars Data

This DAG trains the model based on existing data and puts it in production.
"""

@dag(
    dag_id="train_the_model_with_catboost",
    description="Train the model based on existing data and put it in production",
    doc_md=markdown_text,
    tags=["Train", "Cars"],
    default_args=default_args,
    catchup=False,
)
def train_the_model_with_catboost():

    @task
    def get_or_create_experiment(experiment_name):
        """Get, create, or restore an experiment in MLflow."""
        import mlflow
        import logging
        from mlflow.tracking import MlflowClient
        from airflow.models import Variable

        logger = logging.getLogger("airflow.task")

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment:
            if experiment.lifecycle_stage == "deleted":
                logger.info("Experiment '%s' is deleted. Restoring it.", experiment_name)
                client.restore_experiment(experiment.experiment_id)
            else:
                logger.info("Experiment '%s' found and is active.", experiment_name)
            return experiment.experiment_id

        logger.info("Experiment '%s' does not exist. Creating it.", experiment_name)
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
        from catboost import CatBoostRegressor
        from sklearn.metrics import (
            mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
        )
        from mlflow.models.signature import infer_signature
        from airflow.models import Variable
        from models.data_loader import load_train_test_data

        logger = logging.getLogger("airflow.task")
        logger.info("Experiment id: %s", experiment_id)

        X_train, y_train, X_test, y_test = load_train_test_data()

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        run_name_parent = "best_hyperparam_" + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')

        model = CatBoostRegressor(
            eval_metric='RMSE',
            random_seed=42,
            verbose=0
        )

        params = {
            'iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'bagging_temperature': [0, 0.5, 1]
        }

        catboost = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=10,
            scoring='neg_root_mean_squared_error',
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        catboost.fit(X_train, y_train)
        catboost_best_model = catboost.best_estimator_

        y_pred = np.expm1(catboost_best_model.predict(X_test))
        y_pred_train = np.expm1(catboost_best_model.predict(X_train))
        y_train_recovered = np.expm1(y_train.to_numpy().flatten())
        y_test_recovered = np.expm1(y_test.to_numpy().flatten())

        metrics = {
            "MAE_training": mean_absolute_error(y_train_recovered, y_pred_train),
            "MAE": mean_absolute_error(y_test_recovered, y_pred),
            "RMSE": mean_squared_error(y_test_recovered, y_pred, squared=False),
            "MAPE": mean_absolute_percentage_error(y_test_recovered, y_pred),
            "R2": r2_score(y_test_recovered, y_pred)
        }

        logger.info("Metrics: %s", metrics)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent, nested=True):
            mlflow.log_params(catboost.best_params_)
            mlflow.log_metrics(metrics)
            artifact_path = "model"
            signature = infer_signature(X_train, catboost_best_model.predict(X_train))

            mlflow.catboost.log_model(
                cb_model=catboost_best_model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name="cars_catboost_model_dev",
                input_example=X_train.head()
            )

            model_uri = mlflow.get_artifact_uri(artifact_path)
            logger.info("Model uri: %s", model_uri)

            client = MlflowClient()
            name = "cars_catboost_model_prod"
            desc = "This model predicts selling price for used cars"

            try:
                model_registered = client.get_registered_model(name)
            except mlflow.exceptions.MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    model_registered = None
                else:
                    raise e

            if not model_registered:
                client.create_registered_model(name=name, description=desc)

            tags = catboost_best_model.get_params()
            tags["model"] = type(catboost_best_model).__name__
            tags["mae_training"] = metrics["MAE_training"]
            tags["mae"] = metrics["MAE"]
            tags["rmse"] = metrics["RMSE"]
            tags["mape"] = metrics["MAPE"]
            tags["r2"] = metrics["R2"]

            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            client.set_registered_model_alias(name, "champion", result.version)

    experiment_id = get_or_create_experiment("Cars")
    train_and_log_model(experiment_id)

train_the_model_with_catboost()
