from airflow.decorators import dag, task
from sklearn.linear_model import Ridge
from config.default_args import default_args  # Importa el archivo

markdown_text = """
### Train Ridge Model for Cars Data

This DAG trains ridge model based on new data, tests the previous model, and put in production the new one 
if it performs  better than the old one. It uses the mean absolute error to evaluate the model with the test data.

"""


@dag(
    dag_id="train_ridge_model",
    description="Train simple model based on new data, tests the previous model, and put in production the new one if "
                "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Train", "Cars", "Ridge"],
    default_args=default_args,
    catchup=False,
)
def train_ridge_model():
    @task.virtualenv(
        task_id="train_ridge_challenger_model",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_ridge_challenger_model():
        import datetime
        import logging
        import mlflow
        import awswrangler as wr
        import numpy as np

        from sklearn.linear_model import Ridge
        from sklearn.base import clone
        from sklearn.metrics import mean_absolute_error
        from mlflow.models import infer_signature

        mlflow.set_tracking_uri('http://mlflow:5002')
        logger = logging.getLogger("airflow.task")

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/cars_X_train_processed.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/cars_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/cars_X_test_processed.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/cars_y_test.csv")

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            # Track the experiment
            experiment = mlflow.set_experiment("Cars")

            mlflow.start_run(run_name='Train_Ridge_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                             experiment_id=experiment.experiment_id,
                             tags={"experiment": "train models", "dataset": "Cars"},
                             log_system_metrics=True)

            params = model.get_params()
            params["model"] = type(model).__name__

            mlflow.log_params(params)

            # Save the artifact of the challenger model
            artifact_path = "model"

            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                serialization_format='cloudpickle',
                registered_model_name="cars_ridge_model_dev",
                metadata={"model_data_version": 1}
            )

            # Obtain the model URI
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, mean_absolute_error, model_uri):
            client = mlflow.MlflowClient()
            name = "cars_ridge_model_dev"

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["mean_absolute_error"] = mean_absolute_error

            # Check if the model is already registered, if not, create a new one
            try:
                client.get_registered_model(name)
            except mlflow.exceptions.RestException:
                client.create_registered_model(name)
                logging.info("Registered model %s", name)

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "challenger", result.version)

        logger.info("Create Model")
        model = Ridge(alpha=1)

        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        model.fit(X_train, y_train)

        # Obtain the metric of the model
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        mean_absolute_error = mean_absolute_error(y_test, y_pred)
        logger.info("Mean Absolute Error: %s", mean_absolute_error)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(model, X_train)
        logger.info("Artifact URI: %s", artifact_uri)

        # Record the model
        register_challenger(model, mean_absolute_error, artifact_uri)
        logger.info("Challenger registered!")

    @task.virtualenv(
        task_id="evaluate_ridge_champion_challenge",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def evaluate_ridge_champion_challenge():
        import mlflow
        import logging
        import awswrangler as wr
        import numpy as np

        from sklearn.metrics import mean_absolute_error

        logger = logging.getLogger("airflow.task")
        mlflow.set_tracking_uri('http://mlflow:5002')

        def load_the_model(model_name, alias):
            client = mlflow.MlflowClient()
            try:
                model_data = client.get_model_version_by_alias(model_name, alias)
                model = mlflow.sklearn.load_model(model_data.source)
                return model
            except mlflow.exceptions.RestException as e:
                if "Registered model alias" in str(e) and "not found" in str(e):
                    logger.warning("Warning: Model alias %s not found for %s. Returning None.", alias, model_name)
                    return None
                else:
                    raise

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/cars_X_test_processed.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/cars_y_test.csv")

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Demote the champion
            try:
                client.delete_registered_model_alias(name, "champion")
            except mlflow.exceptions.RestException as e:
                if "Registered model alias champion not found" in str(e):
                    logger.info("No existing champion to demote.")
                else:
                    raise  # Re-raise any unexpected exceptions

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform in champion
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):

            client = mlflow.MlflowClient()

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

        # Load challenger model
        challenger_model = load_the_model("cars_ridge_model_dev", "challenger")
        # Load the dataset
        X_test, y_test = load_the_test_data()

        y_pred_challenger_log = challenger_model.predict(X_test)
        y_pred_challenger = np.expm1(y_pred_challenger_log)
        mean_absolute_error_challenger = mean_absolute_error(y_test, y_pred_challenger)

        experiment = mlflow.set_experiment("Cars")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        # Load champion model
        champion_model = load_the_model("cars_ridge_model_prod", "champion")
        mean_absolute_error_champion = None

        if champion_model:
            y_pred_champion_log = champion_model.predict(X_test)
            y_pred_champion = np.expm1(y_pred_champion_log)
            mean_absolute_error_champion = mean_absolute_error(y_test, y_pred_champion)

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_mean_absolute_error_challenger", mean_absolute_error_challenger)
            if mean_absolute_error_champion:
                mlflow.log_metric("test_mean_absolute_error_champion", mean_absolute_error_champion)

            if not mean_absolute_error_champion or mean_absolute_error_challenger < mean_absolute_error_champion:
                mlflow.log_param("Winner", 'Challenger')
            else:
                mlflow.log_param("Winner", 'Champion')

        name = "cars_ridge_model_prod"
        if not mean_absolute_error_champion or mean_absolute_error_challenger < mean_absolute_error_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)

    train_ridge_challenger_model() >> evaluate_ridge_champion_challenge()


my_dag = train_ridge_model()
