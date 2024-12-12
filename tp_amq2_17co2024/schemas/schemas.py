from pydantic import BaseModel, Field
from typing import Literal

class ModelInput(BaseModel):
    name: str = Field(..., example="Honda City 1.5 GXI")
    year: int = Field(..., example=2004)
    km_driven: int = Field(..., example=110000)
    fuel: Literal["Diesel", "Petrol", "LPG", "CNG"] = Field(..., example="Petrol")
    seller_type: Literal["Dealer", "Individual", "Trustmark Dealer"] = Field(..., example="Individual")
    transmission: Literal["Manual", "Automatic"] = Field(..., example="Manual")
    owner: Literal[
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ] = Field(..., example="Third Owner")
    mileage: str = Field(..., example="12.8 kmpl")
    engine: str = Field(..., example="1493 CC")
    max_power: str = Field(..., example="100 bhp")
    torque: str = Field(..., example="113.1kgm@ 4600rpm")
    seats: int = Field(..., example=5)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Honda City 1.5 GXI",
                "year": 2004,
                "km_driven": 110000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "Third Owner",
                "mileage": "12.8 kmpl",
                "engine": "1493 CC",
                "max_power": "100 bhp",
                "torque": "113.1kgm@ 4600rpm",
                "seats": 5
            }
        }
class ModelOutput(BaseModel):
    selling_price: float
