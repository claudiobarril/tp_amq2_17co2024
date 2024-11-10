from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import IterativeImputer
import pandas as pd


class MultipleIterativeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
        self.imputer = IterativeImputer(random_state=42, max_iter=10, imputation_order='random')

    def fit(self, X, y=None):
        data_for_imputation = X.drop(columns=self.columns_to_drop, axis=1)

        self.imputer.fit(data_for_imputation)

        return self

    def transform(self, X):
        X = X.copy()
        data_for_imputation = X.drop(columns=self.columns_to_drop, axis=1)

        imputed_data = self.imputer.transform(data_for_imputation)
        imputed_df = pd.DataFrame(imputed_data, columns=data_for_imputation.columns, index=X.index)

        X.update(imputed_df)

        return X
