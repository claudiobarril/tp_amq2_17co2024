import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import Annotated

class ModelOutput(BaseModel):
    """
    Output schema for the cars prediction model.

    This class defines the output fields returned by the cars prediction model along with their descriptions
    and possible values.

    :param output: Output of the model. Price of used car.
    """

    output: float = Field(
        description="Output of the model. Price of used car",
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "output": 123,
                }
            ]
        }
    }