import hashlib

import boto3
import mlflow
import logging
import numpy as np
import pandas as pd
import redis

from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing_extensions import Annotated
from fastapi.responses import HTMLResponse

from models.model_input import ModelInput
from models.model_loader import ModelLoader
from models.model_output import ModelOutput
# from pipeline_ import final_pipeline
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from schemas import ModelInput, ModelOutput
from models.prediction_key import PredictionKey
import boto3
from sklearn.experimental import enable_iterative_imputer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuración de boto3 para MinIO
s3_bucket = 'data'
s3_key = 'pipeline/final_pipeline.joblib'
local_path = '/tmp/final_pipeline.joblib'

s3_client = boto3.client('s3',
                         aws_access_key_id='minio',
                         aws_secret_access_key='minio123',
                         endpoint_url='http://s3:9000')

s3_client.download_file(s3_bucket, s3_key, local_path)

final_pipeline = load(local_path)


# Cargar el modelo y el pipeline
# loader = ModelLoader("best_catboost_model", "cars_best_model")
# final_pipeline = load("s3://data/pipeline/final_pipeline.joblib")
r = redis.Redis(host='redis', port=6379, decode_responses=True)


# Instancia de la aplicación FastAPI con descripción
app = FastAPI(
    title="Car Price Predictor API",
    description=(
        "Esta API permite predecir el precio de autos usados mediante un modelo de machine learning. "
        "Envía las características del auto y recibe una predicción del precio."
    ),
    version="1.0.0",
    
)
# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Cargar el modelo al iniciar la aplicación."""
    try:
        # loader.load_model()
        logger.info("Modelo cargado correctamente al iniciar la aplicación.")
    except Exception as e:
        logger.error(f"Error al cargar el modelo durante el inicio: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Root endpoint que devuelve un mensaje de bienvenida con los integrantes del trabajo práctico.
    """
    return """
    <h2>Bienvenido a la API de Predicción de Precios de Autos Usados</h2>
    <p>Esta API utiliza un modelo de machine learning para predecir el precio de autos usados.</p>
    <h3>Integrantes del Trabajo Práctico:</h3>
    <ul>
        <li><strong>Christian</strong>: christian.tpg@gmail.com</li>
        <li><strong>Claudio</strong>: claudiobarril@gmail.com</li>
        <li><strong>Iñaki</strong>: ilarrumbide10@gmail.com</li>
    </ul>
    """


@app.post("/predict/", response_model=ModelOutput)
async def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks,
):
    """
    Endpoint para predecir el precio de un auto usado.

    Recibe las características del auto y devuelve el precio predicho.
    """
    # Verificar si el modelo necesita ser actualizado en segundo plano
    # background_tasks.add_task(loader.check_model)

    try:
        print("caca")
        logger.info('Convertir el input a DataFrame')
        features_df = pd.DataFrame([features.dict()])

        logger.info(features_df)
        logger.info(type(features_df))

        logger.info('Procesar las features con el pipeline')
        features_processed = final_pipeline.transform(features_df)

        logger.info('Crear keys')
        keys, hashes = PredictionKey().from_pipeline(features_processed)

        logger.info('Buscar keys en Redis')
        model_output = r.get(hashes[0])

        logger.info(f'Predecir {hashes[0]}')
        if model_output is None:
            y_pred = "0"
        else:
            y_pred = model_output

        #
        # # Realizar la predicción en un hilo separado
        # prediction = await asyncio.to_thread(loader.model_ml.predict, features_processed)
        # y_pred = np.exp(prediction)
        #
        logger.info(f"Predicción realizada con éxito. [{y_pred}]")
        logger.info(f"key [{keys[0]}] hash[{hashes[0]}]")

        return ModelOutput(output=y_pred)

    except ValueError as e:
        logger.error(f"Error de validación: {e}")
        raise HTTPException(status_code=422, detail=f"Error de validación: {e}")

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Acciones al cerrar la aplicación."""
    logger.info("Cerrando la aplicación. Cancelando tareas pendientes...")
