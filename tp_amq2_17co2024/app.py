import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from models.model_input import ModelInput
from models.model_loader import ModelLoader
from models.model_output import ModelOutput
from pipeline_ import final_pipeline

# Load the model before start
loader = ModelLoader("best_xgboost_model", "cars_best_model")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Heart Disease Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to Cards Price Predictor API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting car price.

    This endpoint receives features related to a used car and predicts the price using a trained model.
    It returns the prediction result in float format.
    """

    features_df = pd.DataFrame([features])
    prediction = loader.model_ml.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Healthy patient"
    if prediction[0] > 0:
        str_pred = "Heart disease detected"

    # Check if the model has changed asynchronously
    background_tasks.add_task(loader.load_model)

    features = pd.DataFrame({
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
    features_processed = final_pipeline.transform(features)
    prediction = loader.model_ml.predict(features_processed)
    y_pred = np.exp(prediction)

    # Return the prediction result
    return ModelOutput(output=y_pred[0])
