"""Models of outfit object."""

import uuid
import pickle
from typing import Dict, List, Optional
from datetime import datetime

from pydantic.fields import Field
from pydantic import UUID4, validator

from app.pkg.models.base import BaseModel

__all__ = [
    "Outfit",
    "UserOutfitClothes",
]


class BaseOutfit(BaseModel):
    """Base outfit model."""

class Transform(BaseOutfit):
    X: Optional[float] = Field(example=0.05)
    Y: Optional[float] = Field(example=0.0)
    Width: Optional[float] = Field(example=480.24)
    Height: Optional[float] = Field(example=512.11)
    Angle: Optional[float] = Field(example=17.35)
    Scale: Optional[float] = Field(example=1.09)

class OutfitFields:
    """Model fields of outfit model."""

    id: UUID4 = Field(
        description="Outfit id.",
        default_factory=uuid.uuid4,
    )
    user_id: Optional[UUID4] = Field(description="Outfit user id.")
    style_id: Optional[UUID4] = Field(description="Outfit style id.")
    try_on_result_id: Optional[UUID4] = Field(description="Try on result id.")
    purpose_ids: List[UUID4] = Field(description="Outfit purpose id.")

    is_generated: bool = Field(
        description="Generated outfit by model or manualy created.",
    )

    name: Optional[str] = Field(description="Outfit name.")
    note: Optional[str] = Field(description="Outfit notes.")
    image: Optional[str] = Field(description="Outfit image in str.")

    created_at: datetime = Field(description="Created at datetime.")
    updated_at: Optional[datetime] = Field(description="Updated at datetime.")

    transforms: Optional[Dict[UUID4, Transform]] = Field(
        description="Outfit transformations in rotating, scales, angles.",
        example={
            "b2795aea-93ec-406f-8d35-af36322776f9": Transform(),
        }
    )
    seasons: Optional[List[str]] = Field(
        description="Outfit seasons.",
        example=["winter", "summer"],
    )
    privacy: str = Field(description="Outfit privacy text.")

    clothes_ids: List[UUID4] = Field(description="Outfit clothes ids.")
    clothes_vectors: List[List[float]] = Field(
        description="Clothes vectors tensors.",
    )

class Outfit(BaseOutfit):
    id: UUID4 = OutfitFields.id
    user_id: Optional[UUID4] = OutfitFields.user_id
    style_id: Optional[UUID4] = OutfitFields.style_id
    try_on_result_id: Optional[UUID4] = OutfitFields.try_on_result_id
    
    created_at: datetime = OutfitFields.created_at
    updated_at: Optional[datetime] = OutfitFields.updated_at

    name: Optional[str] = OutfitFields.name
    note: Optional[str] = OutfitFields.note
    image: Optional[str] = OutfitFields.image
    transforms: Optional[Dict[UUID4, Transform]] = OutfitFields.transforms
    seasons: Optional[List[str]] = OutfitFields.seasons
    privacy: str = OutfitFields.privacy
    
    purpose_ids: List[UUID4] = OutfitFields.purpose_ids
    generated: bool = OutfitFields.is_generated



class UserOutfitClothes(BaseOutfit):
    outfit_id: UUID4 = OutfitFields.id
    user_id: UUID4 = OutfitFields.user_id
    clothes: List[UUID4] = OutfitFields.clothes_ids
    clothes_tensor: List[List[float]] = OutfitFields.clothes_vectors

    @validator("clothes_tensor", pre=True, always=True)
    def convert_bytes_to_tensor(cls, value):
        for i, tensor in enumerate(value):
            if isinstance(tensor, memoryview):
                tensor = tensor.tobytes() # TODO: add to collect response

            if isinstance(tensor, bytes):
                value[i] = pickle.loads(tensor)

        return value