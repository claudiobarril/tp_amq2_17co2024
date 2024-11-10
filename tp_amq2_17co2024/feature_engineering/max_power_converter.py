from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class MaxPowerConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        max_power_non_null = X[X['max_power'].notnull()].copy()
        X['max_power_bhp'] = max_power_non_null['max_power'].str.replace(r'[^0-9.]+', '', regex=True)
        X['max_power_bhp'] = pd.to_numeric(X['max_power_bhp'], errors='coerce')

        return X.drop(columns=['max_power'])
