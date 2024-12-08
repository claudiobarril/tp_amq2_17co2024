from sklearn.base import BaseEstimator, TransformerMixin


class OwnerMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.owner_mapping = {
            'First Owner': 1,
            'Second Owner': 2,
            'Third Owner': 3,
            'Fourth & Above Owner': 4,
            'Test Drive Car': 5
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['owner'] = X['owner'].map(self.owner_mapping)
        return X
