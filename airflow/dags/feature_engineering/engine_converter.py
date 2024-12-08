from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class EngineConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        engine_non_null = X[X['engine'].notnull()].copy()
        X['engine_cc'] = engine_non_null['engine'].str.replace(r'[^0-9.]+', '', regex=True)
        X['engine_cc'] = pd.to_numeric(X['engine_cc'], errors='coerce')

        return X.drop(columns=['engine'])
