from util import *

# Definir pipelines para preprocesamiento
preprocess_pipeline = Pipeline(steps=[
    ('split_name', SplitName()),
    ('impute_mileage', ImputeMileageWithCar()),
    ('Engine_maxpower',EnginePowerTransformer()),
    ('convert_mileage', ConvertMileageToKmpl()),
    ('impute_max_power', ImputeMaxPowerWithCar()),
    ('standardize_torque', StandardizeTorque()),
])

categorical_cols = ['fuel', 'seller_type', 'transmission', 'make']
columns_to_drop_for_imputation = []

full_pipeline = Pipeline(steps=[
    ('map_owner', MapOwner()),
    ('one_hot_encode', OneHotEncodeCategoricals(categorical_cols=categorical_cols)),
    ('iterative_imputation', IterativeImputation(columns_to_drop=columns_to_drop_for_imputation)),
    ('round_seats', RoundSeats()),
    ('scaler', StandardScaler())  # Agregar el StandardScaler aquí
])

# Pipeline final con preprocesamiento de datos y feature engineering
final_pipeline = Pipeline(steps=[
    ('preprocess', preprocess_pipeline),
    ('features', full_pipeline)
    
])
