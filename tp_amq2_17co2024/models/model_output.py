from pydantic import BaseModel, Field

class ModelOutput(BaseModel):
    """
    Output schema for the cars prediction model.

    This class defines the output fields returned by the cars prediction model along with their descriptions
    and possible values.

    :param selling_price: Output of the model. Price of used car.
    """

    selling_price: float = Field(
        description="Output of the model. Price of used car",
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "selling_price": 123,
                }
            ]
        }
    }
