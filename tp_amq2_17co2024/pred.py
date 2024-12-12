import wr
from catboost import CatBoostRegressor
import pandas as pd
import joblib
import numpy as np
import sys
import os
import boto3
from joblib import load

from models.prediction_key import PredictionKey

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cargar el modelo CatBoost entrenado
# loaded_model = CatBoostRegressor()
# loaded_model.load_model("../../models/catboost_model.cbm")

import pickle
from xgboost import XGBRegressor

with open("../models/best_catboost_model.pkl", "rb") as f:
    model = pickle.load(f)

model.save_model("../models/best_catboost_model.json")
loaded_model = CatBoostRegressor()
loaded_model.load_model("../models/best_catboost_model.json")

s3_bucket = 'data'
s3_key = 'pipeline/final_pipeline.joblib'
local_path = '/tmp/final_pipeline.joblib'

s3_client = boto3.client('s3',
                         aws_access_key_id='minio',
                         aws_secret_access_key='minio123',
                         endpoint_url='http://localhost:9000')

s3_client.download_file(s3_bucket, s3_key, local_path)

final_pipeline = load(local_path)

# def make_predictions(nuevo_dato):
#     nuevo_dato_procesado = final_pipeline.fit_transform(nuevo_dato)
#     return loaded_model.predict(nuevo_dato_procesado)

# Nuevo dato
nuevo_dato = pd.DataFrame({
    'name': ["Toyota Innova 2.5 E 7 STR"],
    'year': [2009],
    'km_driven': [250000],
    'fuel': ['Diesel'],
    'seller_type': ['Individual'],
    'transmission': ['Manual'],
    'owner': ['First Owner'],
    'mileage': ['12.8 kmpl'],
    'engine': ['2494 CC'],
    'max_power': ['102 bhp'],
    'torque': ['20.4@ 1400-3400(kgm@ rpm)'],
    'seats': [7.0]
})

key2, hashes2 = PredictionKey().from_pipeline(final_pipeline.transform(nuevo_dato))
print(f'prediction hash [{hashes2[0]}] key[{key2[0]}]')

# y_pred = np.exp(make_predictions(nuevo_dato))
# print(f"Precio predicho: {y_pred[0]:.2f}")