import boto3
import mlflow
import logging
import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing_extensions import Annotated
from fastapi.responses import HTMLResponse

from models.model_input import ModelInput
from models.model_loader import ModelLoader
from models.model_output import ModelOutput
from pipeline_ import final_pipeline
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from schemas import ModelInput, ModelOutput

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el modelo y el pipeline
loader = ModelLoader("best_xgboost_model", "cars_best_model")
final_pipeline = load("../models/final_pipeline.joblib")

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
        loader.load_model()
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
    background_tasks.add_task(loader.check_model)

    try:
        # Convertir el input a DataFrame
        features_df = pd.DataFrame([features.dict()])

        # Procesar las features con el pipeline
        features_processed = final_pipeline.transform(features_df)

        # Realizar la predicción en un hilo separado
        prediction = await asyncio.to_thread(loader.model_ml.predict, features_processed)
        y_pred = np.exp(prediction)

        logger.info("Predicción realizada con éxito.")
        return ModelOutput(output=float(y_pred[0]))

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
