import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import re
import os


# Transformadores personalizados
class SplitName(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[['make', 'model']] = X['name'].str.split(' ', n=1, expand=True)
        
        return X

class ImputeMileageWithCar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for idx, row in df[df['mileage'] == '0.0 kmpl'].iterrows():
            similar_condition = (df['name'] == row['name']) & (df['year'] == row['year']) & (df['fuel'] == row['fuel'])
            similar_vehicles = df[similar_condition]
            if not similar_vehicles.empty:
                df.at[idx, 'mileage'] = similar_vehicles['mileage'].iloc[0]
        return df

class ConvertMileageToKmpl(BaseEstimator, TransformerMixin):
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

class ImputeMaxPowerWithCar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        for idx, row in df[df['max_power'].isin(['bhp', '0'])].iterrows():
            similar_condition = (df['name'] == row['name']) & (df['year'] == row['year']) & (df['fuel'] == row['fuel'])
            similar_vehicles = df[similar_condition]
            if not similar_vehicles.empty:
                df.at[idx, 'max_power'] = similar_vehicles['max_power'].iloc[0]
        return df
    
class EnginePowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Process engine column
        engine_non_null = X[X['engine'].notnull()].copy()
        X['engine_cc'] = engine_non_null['engine'].str.replace(r'[^0-9.]+', '', regex=True)
        X['engine_cc'] = pd.to_numeric(X['engine_cc'], errors='coerce')

        # Process max_power column
        max_power_non_null = X[X['max_power'].notnull()].copy()
        X['max_power_bhp'] = max_power_non_null['max_power'].str.replace(r'[^0-9.]+', '', regex=True)
        X['max_power_bhp'] = pd.to_numeric(X['max_power_bhp'], errors='coerce')

        

        return X

class StandardizeTorque(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def standardize_torque(torque_str):
            if pd.isna(torque_str):
                return {'torque_peak_power': np.nan, 'torque_peak_speed': np.nan}

            patterns = [
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*at\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*@\s*([-\d\s,]+)\s*\(kgm@\s*rpm\)",
                r"(\d*\.?\d+)\s*kgm\s*at\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*\((\d*\.?\d+)\s*kgm\)\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*\((\d*\.?\d+)\)\s*@\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*(kgm|nm)?\s*/\s*([-\d\s,]+)\s*(rpm)?",
                r"(\d*\.?\d+)\s*/\s*([-\d\s,]+)"
            ]

            torque_peak_power = None
            torque_peak_speed = None

            for pattern in patterns:
                match = re.findall(pattern, str(torque_str).lower())
                if match:
                    if len(match[0]) == 4:
                        value, unit, rpm_range, _ = match[0]
                    elif len(match[0]) == 3:
                        value, unit, rpm_range = match[0]
                    elif len(match[0]) == 2:
                        value, rpm_range = match[0]
                        unit = None
                    else:
                        continue

                    value = float(value)

                    if 'kgm' in str(torque_str).lower() or (unit and 'kgm' in unit):
                        value *= 9.81

                    torque_peak_power = value

                    if rpm_range:
                        rpm_range = rpm_range.replace(',', '')
                        if '-' in rpm_range:
                            rpm_values = list(map(int, rpm_range.split('-')))
                            torque_peak_speed = max(rpm_values)
                        else:
                            torque_peak_speed = int(rpm_range.strip())

                    break
            return {'torque_peak_power': torque_peak_power, 'torque_peak_speed': torque_peak_speed}
        
        X = X.copy()
        torque_results = X['torque'].apply(standardize_torque)
        X['torque_peak_power'] = torque_results.apply(lambda x: x['torque_peak_power'])
        X['torque_peak_speed'] = torque_results.apply(lambda x: x['torque_peak_speed'])
        X = X.drop(columns=['name',  'torque','engine', 'max_power','model'])
        return X

class MapOwner(BaseEstimator, TransformerMixin):
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

class RoundSeats(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.final_columns_ = None  # Cambiado de columns_ a final_columns_

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Redondear las columnas 'seats'
        X['seats'] = X['seats'].round().astype(int)
        # Guardar las columnas
        self.final_columns_ = X.columns  # Cambiado de columns_ a final_columns_
        return X
    
    

class OneHotEncodeCategoricals(BaseEstimator, TransformerMixin):
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

class IterativeImputation(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop else []
        self.imputer = IterativeImputer(random_state=42, max_iter=10, imputation_order='random')
    
    def fit(self, X, y=None):
        
        data_for_imputation = X.drop(columns=self.columns_to_drop)
        

        self.imputer.fit(data_for_imputation)
        
        return self
    
    def transform(self, X):
        X = X.copy()
        data_for_imputation = X.drop(columns=self.columns_to_drop)
        
        imputed_data = self.imputer.transform(data_for_imputation)
        imputed_df = pd.DataFrame(imputed_data, columns=data_for_imputation.columns, index=X.index)
        
        X.update(imputed_df)

        
        
        return X