import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing_extensions import Annotated

from models.model_input import ModelInput
from models.model_loader import ModelLoader
from models.model_output import ModelOutput
from pipeline_ import final_pipeline
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from schemas import ModelInput, ModelOutput
# Load the model before start
loader = ModelLoader("best_xgboost_model", "cars_best_model")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes desde cualquier origen (puedes restringirlo a dominios específicos)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)


@app.get("/")
async def read_root():
    """
    Root endpoint of the Heart Disease Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to Cards Price Predictor API"}))


final_pipeline = load('../models/final_pipeline.joblib')

@app.on_event("startup")
def startup_event():
    loader.load_model()


@app.post("/predict/", response_model=ModelOutput)
async def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting car price.

    Este endpoint recibe las características de un auto usado y predice su precio utilizando un modelo entrenado.
    """
    # Verificar si el modelo necesita ser actualizado en segundo plano
    background_tasks.add_task(loader.check_model)

    # Convertir el input a DataFrame
    features_df = pd.DataFrame([features.dict()])

    # Procesar las features con el pipeline
    features_processed = final_pipeline.transform(features_df)

    # Realizar la predicción en un hilo separado para no bloquear el bucle de eventos
    try:
        prediction = await asyncio.to_thread(loader.model_ml.predict, features_processed)
    except asyncio.CancelledError:
        print("La tarea fue cancelada.")
        raise

    y_pred = np.exp(prediction)

    # Retornar el resultado
    return ModelOutput(output=float(y_pred[0]))


@app.on_event("shutdown")
async def shutdown_event():
    print("Cerrando la aplicación. Cancelando tareas pendientes...")

