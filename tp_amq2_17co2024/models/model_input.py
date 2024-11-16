from typing import Literal
from pydantic import BaseModel, Field


class ModelInput(BaseModel):
    """
    Input schema for the car dataset.

    This class defines the input fields required for car data along with their descriptions
    and validation constraints.

    :param brand: Brand of the car.
    :param model: Model of the car.
    :param year: year of the sell.
    :param km_driven: Total kilometers driven by the car.
    :param fuel: Type of fuel used by the car.
    :param seller_type: Type of seller. Can be 'individual', 'dealer', or 'trustmark dealer'.
    :param transmission: Type of transmission. Can be 'manual' or 'automatic'.
    :param owner: Ownership type. Can be 'first', 'second', 'third', or 'fourth & above'.
    :param mileage: Mileage of the car in kmpl.
    :param engine: Engine capacity in cc.
    :param max_power: Maximum power of the car in bhp.
    :param torque_peak_power: Torque peak power of the car.
    :param torque_peak_speed: Torque peak speed of the car in rpm.
    :param seats: Number of seats in the car.
    """

    brand: str = Field(description="Brand of the car")
    model: str = Field(description="Model of the car")
    year: int = Field(
        description="Year of the sell",
        ge=1886,  # First car ever made
        le=2024,  # Current year
    )
    km_driven: int = Field(
        description="Total kilometers driven by the car",
        ge=0,
    )
    fuel: Literal["petrol", "diesel", "cng", "lpg", "electric"] = Field(
        description="Type of fuel used by the car"
    )
    seller_type: Literal["individual", "dealer", "trustmark dealer"] = Field(
        description="Type of seller"
    )
    transmission: Literal["manual", "automatic"] = Field(
        description="Type of transmission"
    )
    owner: Literal["first", "second", "third", "fourth & above"] = Field(
        description="Ownership type"
    )
    mileage: float = Field(
        description="Mileage of the car in kmpl",
        ge=0.0,
    )
    engine: int = Field(
        description="Engine capacity in cc",
        ge=50,  # Smallest engine cars
        le=10000,  # Largest engine in consumer cars
    )
    max_power: float = Field(
        description="Maximum power of the car in bhp",
        ge=0.0,
    )
    torque_peak_power: float = Field(
        description="Torque peak power of the car",
        ge=0.0,
    )
    torque_peak_speed: int = Field(
        description="Torque peak speed of the car in rpm",
        ge=0,
    )
    seats: int = Field(
        description="Number of seats in the car",
        ge=2,
        le=12,  # Typical range for cars
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "Toyota",
                    "model": "Corolla",
                    "year": 2020,
                    "km_driven": 15000,
                    "fuel": "petrol",
                    "seller_type": "dealer",
                    "transmission": "manual",
                    "owner": "first",
                    "mileage": 16.7,
                    "engine": 1800,
                    "max_power": 140.0,
                    "torque_peak_power": 170.0,
                    "torque_peak_speed": 4000,
                    "seats": 5,
                }
            ]
        }
    }
