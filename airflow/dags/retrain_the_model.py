from airflow.decorators import dag, task
from airflow.models import Variable
from config.default_args import default_args

markdown_text = """
### Re-Train the Model for Cars Data

This DAG re-trains the model based on new data, tests the previous model, and put in production the new one 
if it performs  better than the old one. It uses the MAE score to evaluate the model with the test data.

"""

@dag(
    dag_id="retrain_the_model",
    description="Re-train the model based on new data, tests the previous model, and put in production the new one if "
                "it performs better than the old one",
    doc_md=markdown_text,
    tags=["Re-Train", "Cars"],
    default_args=default_args,
    catchup=False,
)
def processing_dag():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def train_the_challenger_model():
        import datetime
        import mlflow
        import awswrangler as wr
        import numpy as np

        from xgboost import XGBClassifier
        from sklearn.metrics import mean_squared_error
        from mlflow.models import infer_signature
        from airflow.models import Variable

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        def load_the_champion_model():

            model_name = "cars_model_prod"
            alias = "champion"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            champion_version = mlflow.xgboost.load_model(model_data.source)

            return champion_version

        def load_the_train_test_data():
            X_train = wr.s3.read_csv(Variable.get("cars_X_train_processed_location"))
            y_train = wr.s3.read_csv(Variable.get("cars_y_train_location"))
            X_test = wr.s3.read_csv(Variable.get("cars_X_test_processed_location"))
            y_test = wr.s3.read_csv(Variable.get("cars_y_test_location"))

            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):

            # Track the experiment
            experiment = mlflow.set_experiment("Cars")

            mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
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
                registered_model_name="cars_model_dev",
                metadata={"model_data_version": 1}
            )

            # Obtain the model URI
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, neg_mse, model_uri):

            client = mlflow.MlflowClient()
            name = "cars_model_prod"

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["neg_MSE"] = neg_mse

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "challenger", result.version)

        # Load the champion model
        champion_model = load_the_champion_model()

        # Clone the model
        params = champion_model.get_params()
        challenger_model = type(champion_model)(**params)

        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        # Fit the training model
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred_log = challenger_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        neg_mse = -mean_squared_error(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Record the model
        register_challenger(challenger_model, neg_mse, artifact_uri)


    @task.virtualenv(
        task_id="evaluate_champion_challenge",
        requirements=["scikit-learn==1.3.2",
                      "mlflow==2.10.2",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr
        import numpy as np

        from sklearn.metrics import mean_squared_error
        from airflow.models import Variable

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        def load_the_model(alias):
            model_name = "cars_model_prod"

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            model = mlflow.xgboost.load_model(model_data.source)

            return model

        def load_the_test_data():
            X_test = wr.s3.read_csv(Variable.get("cars_X_test_processed_location"))
            y_test = wr.s3.read_csv(Variable.get("cars_y_test_location"))

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Demote the champion
            client.delete_registered_model_alias(name, "champion")

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

        # Load the champion model
        champion_model = load_the_model("champion")

        # Load the challenger model
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        y_pred_champion_log = champion_model.predict(X_test)
        y_pred_champion = np.expm1(y_pred_champion_log)
        neg_mse_champion = -mean_squared_error(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger_log = challenger_model.predict(X_test)
        y_pred_challenger = np.expm1(y_pred_challenger_log)
        neg_mse_challenger = -mean_squared_error(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment("Cars")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_neg_mse_challenger", neg_mse_challenger)
            mlflow.log_metric("test_neg_mse_champion", neg_mse_champion)

            if neg_mse_challenger > neg_mse_champion:
                mlflow.log_param("Winner", 'Challenger')
                promote_challenger("cars_model_prod")
            else:
                mlflow.log_param("Winner", 'Champion')
                demote_challenger("cars_model_prod")

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
