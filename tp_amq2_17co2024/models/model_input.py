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

# TODO: change all of it
class ModelInput(BaseModel):
    """
    Input schema for the cards prediction model.

    This class defines the input fields required by the heart disease prediction model along with their descriptions
    and validation constraints.

    :param age: Age of the patient (0 to 150).
    :param sex: Sex of the patient. 1: male; 0: female.
    :param cp: Chest pain type. 1: typical angina; 2: atypical angina; 3: non-anginal pain; 4: asymptomatic.
    :param trestbps: Resting blood pressure in mm Hg on admission to the hospital (90 to 220).
    :param chol: Serum cholestoral in mg/dl (110 to 600).
    :param fbs: Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl.
    :param restecg: Resting electrocardiographic results. 0: normal; 1: having ST-T wave abnormality; 2: showing
                    probable or definite left ventricular hypertrophy.
    :param thalach: Maximum heart rate achieved (beats per minute) (50 to 210).
    :param exang: Exercise induced angina. 1: yes; 0: no.
    :param oldpeak: ST depression induced by exercise relative to rest (0.0 to 7.0).
    :param slope: The slope of the peak exercise ST segment. 1: upsloping; 2: flat; 3: downsloping.
    :param ca: Number of major vessels colored by flourosopy (0 to 3).
    :param thal: Thalassemia disease. 3: normal; 6: fixed defect; 7: reversable defect.
    """

    age: int = Field(
        description="Age of the patient",
        ge=0,
        le=150,
    )
    sex: int = Field(
        description="Sex of the patient. 1: male; 0: female",
        ge=0,
        le=1,
    )
    cp: int = Field(
        description="Chest pain type. 1: typical angina; 2: atypical angina, 3: non-anginal pain; 4: asymptomatic",
        ge=1,
        le=4,
    )
    trestbps: float = Field(
        description="Resting blood pressure in mm Hg on admission to the hospital",
        ge=90,
        le=220,
    )
    chol: float = Field(
        description="Serum cholestoral in mg/dl",
        ge=110,
        le=600,
    )
    fbs: int = Field(
        description="Fasting blood sugar. 1: >120 mg/dl; 0: <120 mg/dl",
        ge=0,
        le=1,
    )
    restecg: int = Field(
        description="Resting electrocardiographic results. 0: normal; 1:  having ST-T wave abnormality (T wave "
                    "inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite "
                    "left ventricular hypertrophy by Estes' criteria",
        ge=0,
        le=2,
    )
    thalach: float = Field(
        description="Maximum heart rate achieved (beats per minute)",
        ge=50,
        le=210,
    )
    exang: int = Field(
        description="Exercise induced angina. 1: yes; 0: no",
        ge=0,
        le=1,
    )
    oldpeak: float = Field(
        description="ST depression induced by exercise relative to rest",
        ge=0.0,
        le=7.0,
    )
    slope: int = Field(
        description="The slope of the peak exercise ST segment .1: upsloping; 2: flat, 3: downsloping",
        ge=1,
        le=3,
    )
    ca: int = Field(
        description="Number of major vessels colored by flourosopy",
        ge=0,
        le=3,
    )
    thal: Literal[3, 6, 7] = Field(
        description="Thalassemia disease. 3: normal; 6: fixed defect; 7: reversable defect",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 67,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 160.0,
                    "chol": 286.0,
                    "fbs": 0,
                    "restecg": 2,
                    "thalach": 108.0,
                    "exang": 1,
                    "oldpeak": 1.5,
                    "slope": 2,
                    "ca": 3,
                    "thal": 3,
                }
            ]
        }
    }