from sklearn.base import BaseEstimator, TransformerMixin

class NameSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[['make', 'model']] = X['name'].str.split(' ', n=1, expand=True)

        return X.drop(columns=['name'])
