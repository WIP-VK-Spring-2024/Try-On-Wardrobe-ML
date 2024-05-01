"""Models of clothes vector object."""

import uuid
from typing import List
import pickle

from pydantic.fields import Field
from pydantic import UUID4, validator

from app.pkg.models.base import BaseModel

__all__ = [
    "ClothesVector",
    "ClothesVectorCreateCmd",
]


class BaseClothesVector(BaseModel):
    """Base clothes vector model."""

class ClothesVectorFields:
    """Model fields of cut model."""

    id: UUID4 = Field(description="Clothes vector id.")
    clothes_id: UUID4 = Field(description="Clothes id.", default_factory=uuid.uuid4)
    tensor: List[float] = Field(
        description="Clothes vector tensor.",
        example=[
            0.280,
            0.023,
            0.989,
            0.453,
        ],
    )
    tensor_bytes: bytes = Field(
        description="Clothes vecotr tensor in bytes format.",
    )

class ClothesVector(BaseClothesVector):
    id: UUID4 = ClothesVectorFields.id
    clothes_id: UUID4 = ClothesVectorFields.clothes_id
    tensor: List[float] = ClothesVectorFields.tensor

    @validator("tensor", pre=True, always=True)
    def convert_bytes_to_tensor(cls, value):
        if isinstance(value, bytes):
            return pickle.loads(value)
        return value

    @classmethod
    def from_orm(cls, data):
        tensor = data.__dict__["tensor"]
        data.__dict__["tensor"] = pickle.loads(tensor)
        return super().from_orm(data)

    class Config:
        orm_mode = True

class ClothesVectorCreateCmd(BaseClothesVector):
    id: UUID4 = ClothesVectorFields.id
    clothes_id: UUID4 = ClothesVectorFields.clothes_id
    tensor: bytes = ClothesVectorFields.tensor_bytes

    @validator("tensor", pre=True, always=True)
    def convert_tensor_to_bytes(cls, value):
        if value is not None:
            return pickle.dumps(value)
        return value
