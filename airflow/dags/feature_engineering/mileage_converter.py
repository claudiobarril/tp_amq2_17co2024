from sklearn.base import BaseEstimator, TransformerMixin
import re


class MileageConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['mileage'] = X['mileage'].fillna('').astype(str)

        def convert_to_kmpl(row):
            kmpl_rx = r"(\d+\.?\d*)\s*kmpl"
            kmkg_rx = r"(\d+\.?\d*)\s*km/kg"
            conversion_factor = 1.39
            mileage_value = None

            if row['fuel'].lower() in ['diesel', 'petrol']:
                match = re.search(kmpl_rx, row['mileage'])
                if match:
                    mileage_value = float(match.group(1))
            elif row['fuel'].lower() in ['cng', 'lpg']:
                match = re.search(kmkg_rx, row['mileage'])
                if match:
                    kmkg_value = float(match.group(1))
                    mileage_value = kmkg_value * conversion_factor

            return mileage_value

        X['mileage_kmpl'] = X.apply(convert_to_kmpl, axis=1)
        return X.drop(columns=['mileage'])
