from airflow.decorators import dag, task
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
        import mlflow
        import awswrangler as wr
        import numpy as np

        from xgboost import XGBClassifier
        from sklearn.metrics import mean_squared_error
        from airflow.models import Variable
        from models.model_manager import ModelManager
        from experiment.experiment_tracker import ExperimentTracker

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model_manager = ModelManager()
        experiment_tracker = ExperimentTracker()

        def load_train_test_data():
            X_train = wr.s3.read_csv(Variable.get("cars_X_train_processed_location"))
            y_train = wr.s3.read_csv(Variable.get("cars_y_train_location"))
            X_test = wr.s3.read_csv(Variable.get("cars_X_test_processed_location"))
            y_test = wr.s3.read_csv(Variable.get("cars_y_test_location"))
            return X_train, y_train, X_test, y_test

        # Load the champion model
        champion_model = model_manager.load_model("champion")

        # Clone the model
        params = champion_model.get_params()
        challenger_model = type(champion_model)(**params)

        # Load the dataset
        X_train, y_train, X_test, y_test = load_train_test_data()

        # Fit the training model
        challenger_model.fit(X_train, y_train.to_numpy().ravel())

        # Obtain the metric of the model
        y_pred_log = challenger_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        neg_mse = -mean_squared_error(y_test.to_numpy().ravel(), y_pred)

        # Track the experiment
        artifact_uri = experiment_tracker.track_experiment(challenger_model, "cars_model_dev", X_train)

        # Record the model
        model_manager.register_challenger(challenger_model, neg_mse, artifact_uri)

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
        from models.model_manager import ModelManager

        mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        model_manager = ModelManager()

        def load_test_data():
            X_test = wr.s3.read_csv(Variable.get("cars_X_test_processed_location"))
            y_test = wr.s3.read_csv(Variable.get("cars_y_test_location"))
            return X_test, y_test

        # Load the champion and challenger models
        champion_model = model_manager.load_model("champion")
        challenger_model = model_manager.load_model("challenger")

        # Load the dataset
        X_test, y_test = load_test_data()

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
                model_manager.promote_challenger()
            else:
                mlflow.log_param("Winner", 'Champion')
                model_manager.demote_challenger()

    train_the_challenger_model() >> evaluate_champion_challenge()


my_dag = processing_dag()
