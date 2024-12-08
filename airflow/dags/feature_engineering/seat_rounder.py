from sklearn.base import BaseEstimator, TransformerMixin


class SeatRounder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Redondear las columnas 'seats'
        X['seats'] = X['seats'].round().astype(int)
        # Aquí devolvemos el DataFrame y también guardamos las columnas finales para referencia
        self.final_columns_ = X.columns  # Guardamos las columnas finales
        return X
