from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer
import pandas as pd
import logging


class MultipleIterativeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
        self.imputer = IterativeImputer(random_state=42, max_iter=10, imputation_order='random', keep_empty_features=True)

    def fit(self, X, y=None):
        data_for_imputation = X.drop(columns=self.columns_to_drop, axis=1)

        self.imputer.fit(data_for_imputation)

        return self

    def transform(self, X):
        logger = logging.getLogger("airflow.task")

        X = X.copy()
        logger.info("Shape of X: %s", X.shape)
        data_for_imputation = X.drop(columns=self.columns_to_drop, axis=1)
        missing_columns = data_for_imputation.columns[data_for_imputation.isnull().all()]
        logger.info("Missing columns: %s", missing_columns)
        logger.info("[ANTES] Columnas del dataset data_for_imputation: %s", data_for_imputation.columns)
        imputed_data = self.imputer.transform(data_for_imputation)
        logger.info("Shape of data_for_imputation: %s", data_for_imputation.shape)
        logger.info("Shape of imputed_data: %s", imputed_data.shape)
        logger.info("Columns to drop: %s", self.columns_to_drop)
        logger.info("Remaining columns after dropping: %s", data_for_imputation.columns)
        logger.info("Index of X: %s", X.index)

        imputed_df = pd.DataFrame(imputed_data, columns=data_for_imputation.columns, index=X.index)

        X.update(imputed_df)

        return X
