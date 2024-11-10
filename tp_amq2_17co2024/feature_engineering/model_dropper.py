from sklearn.base import BaseEstimator, TransformerMixin

class ModelDropper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=['model'])
