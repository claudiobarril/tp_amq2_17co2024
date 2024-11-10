import pandas as pd
from feature_engineering.name_splitter import NameSplitter
from feature_engineering.owner_mapper import OwnerMapper
from feature_engineering.seat_rounder import SeatRounder
from feature_engineering.max_power_converter import MaxPowerConverter
from feature_engineering.engine_converter import EngineConverter
from feature_engineering.multiple_interative_imputer import MultipleIterativeImputer
from feature_engineering.multiple_one_hot_encoder import MultipleOneHotEncoder
from feature_engineering.torque_standardizer import TorqueStandardizer
from feature_engineering.mileage_converter import MileageConverter
from feature_engineering.model_dropper import ModelDropper
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler


class CarsPipeline(Pipeline):
    def __init__(self):
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'make']

        preprocess_pipeline = Pipeline(steps=[
            ('name_spliter', NameSplitter()),
            ('mileage_converter', MileageConverter()),
            ('engine_converter', EngineConverter()),
            ('max_power_converter', MaxPowerConverter()),
            ('torque_standardizer', TorqueStandardizer()),
            ('map_owner', OwnerMapper()),
            ('model_dropper', ModelDropper()),
            ('multiple_one_hot_encoder', MultipleOneHotEncoder(categorical_cols=categorical_cols)),
        ])

        columns_to_drop_for_imputation = []

        full_pipeline = Pipeline(steps=[
            ('multiple_iterative_imputer', MultipleIterativeImputer(columns_to_drop=columns_to_drop_for_imputation)),
            ('round_seats', SeatRounder()),
        ])

        final_pipeline = Pipeline(steps=[
            ('preprocess_pipeline', preprocess_pipeline),
            ('full_pipeline', full_pipeline),
        ])

        self.pipeline = Pipeline(steps=[
            ('final_pipeline', final_pipeline),
            ('scaler', StandardScaler())
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)

    def transform(self, X, y=None):
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X, y)

    def final_columns(self):
        return self.pipeline.named_steps['final_pipeline'].named_steps['full_pipeline'].named_steps['round_seats'].final_columns_

    def fit_transform_df(self, X, y=None):
        processed = self.pipeline.fit_transform(X, y)
        return pd.DataFrame(processed, columns=self.final_columns())

    def transform_df(self, X):
        processed = self.pipeline.transform(X)
        return pd.DataFrame(processed, columns=self.final_columns())
