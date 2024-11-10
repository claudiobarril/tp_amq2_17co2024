from catboost import CatBoostRegressor
import pandas as pd
import joblib
import numpy as np
import sys
import os

# Subir un nivel desde 'models' y apuntar a la carpeta 'src' que contiene 'util.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora puedes importar util
import util

# Cargar el modelo CatBoost entrenado
catboost_model = CatBoostRegressor()
catboost_model.load_model("../../catboost_model.cbm")

final_pipeline = joblib.load('../../final_pipeline.joblib')

def make_predictions(nuevo_dato):
    nuevo_dato_procesado = final_pipeline.transform(nuevo_dato)
    return catboost_model.predict(nuevo_dato_procesado)

# Nuevo dato
nuevo_dato = pd.DataFrame({
    'name': ['Honda City 1.5 GXI'], 
    'year': [2004], 
    'km_driven': [110000], 
    'fuel': ['Petrol'], 
    'seller_type': ['Individual'], 
    'transmission': ['Manual'], 
    'owner': ['Third Owner'], 
    'mileage': ['12.8 kmpl'], 
    'engine': ['1493 CC'], 
    'max_power': ['100 bhp'], 
    'torque': ['113.1kgm@ 4600rpm'], 
    'seats': [5]
})

# Realizar la predicción
y_pred = np.exp(make_predictions(nuevo_dato))
print(f"Precio predicho: {y_pred[0]:.2f}")

