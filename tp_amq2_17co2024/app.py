import asyncio
import logging
import os
import boto3
import numpy as np
import pandas as pd
import redis
import batch_prediction

from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from joblib import load
from typing_extensions import Annotated
from dotenv import load_dotenv

from models.model_loader import ModelLoader
from models.model_output import ModelOutput
from models.prediction_key import PredictionKey
from schemas import ModelInput
from sklearn.experimental import enable_iterative_imputer

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración global
S3_BUCKET = os.getenv("S3_BUCKET", "data")
PIPELINE_OBJECT_KEY = os.getenv("PIPELINE_OBJECT_KEY", "pipeline/final_pipeline.joblib")
PIPELINE_DOWNLOAD_PATH = "/tmp/final_pipeline.joblib"
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Inicialización de recursos globales
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('MINIO_SECRET_ACCESS_KEY'),
    endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL')
)
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
final_pipeline = None
loader = ModelLoader("cars_model_prod", "champion")

# Instancia de FastAPI
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

def load_pipeline():
    """Descarga y carga el pipeline desde S3."""
    global final_pipeline
    try:
        s3_client.download_file(S3_BUCKET, PIPELINE_OBJECT_KEY, PIPELINE_DOWNLOAD_PATH)
        final_pipeline = load(PIPELINE_DOWNLOAD_PATH)
        logger.info("Pipeline cargado correctamente.")
    except Exception as e:
        logger.error(f"Error al cargar el pipeline: {e}")
        raise RuntimeError(f"Error al cargar el pipeline: {e}")


@app.on_event("startup")
async def startup_event():
    """Inicializa dependencias y carga recursos al iniciar la aplicación."""
    try:
        # Cargar el pipeline
        logger.info("Cargando pipeline...")
        await asyncio.to_thread(load_pipeline)
        logger.info("Pipeline cargado correctamente.")
        # Cargar el modelo
        logger.info("Cargando modelo...")
        await asyncio.to_thread(loader.load_model)
        logger.info("Modelo cargado correctamente.")
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {e}")
        raise e


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


# noinspection PyAsyncCall
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
    background_tasks.add_task(loader.check_model)

    try:
        logger.info('Convertir el input a DataFrame')
        features_df = pd.DataFrame([features.dict()])

        logger.info('Procesar las features con el pipeline')
        features_processed = final_pipeline.transform(features_df)

        logger.info('Crear keys')
        keys, hashes = PredictionKey().from_pipeline(features_processed)

        logger.info('Buscar keys en Redis')
        model_output = r.get(hashes[0])

        if model_output is None:
            logger.info(f"No existe predicción para {hashes[0]}. Ejecutando predicción.")
            pred = await asyncio.to_thread(loader.model_ml.predict, features_processed)
            logger.info(f"Guardando solicitud para futuras predicciones.")
            asyncio.create_task(
                asyncio.to_thread(batch_prediction.add_request, s3_client, features_df)
            )
            y_pred = np.exp(pred)
        else:
            y_pred = model_output

        logger.info(f"Predicción realizada con éxito: {y_pred}")
        return ModelOutput(selling_price=y_pred)

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
