import awswrangler as wr
from airflow.models import Variable


def load_train_test_data():
    X_train = wr.s3.read_csv(Variable.get("cars_X_train_processed_location"))
    y_train = wr.s3.read_csv(Variable.get("cars_y_train_location"))
    X_test = wr.s3.read_csv(Variable.get("cars_X_test_processed_location"))
    y_test = wr.s3.read_csv(Variable.get("cars_y_test_location"))
    return X_train, y_train, X_test, y_test
