from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd


class MultipleOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        # Añadimos handle_unknown='ignore' para evitar errores con categorías no vistas en el test
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        X = X.copy()

        encoded_columns = self.encoder.transform(X[self.categorical_cols])
        encoded_df = pd.DataFrame(encoded_columns, columns=self.encoder.get_feature_names_out(self.categorical_cols))
        encoded_df.index = X.index
        X = pd.concat([X.drop(columns=self.categorical_cols), encoded_df], axis=1)

        return X