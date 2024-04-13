"""Models of outfit gen model."""

import uuid
from typing import List

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings
from app.pkg.models.app.image_category import ImageCategory

__all__ = [
    "OutfitGenClothes",
    "OutfitGenTaskCmd",
    "OutfitGenResponseCmd",
]


class BaseOutfitGenModel(BaseModel):
    """Base model for outfit gen model."""


class OutfitGenFields:
    """Model fields of outfit gen model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    prompt: str = Field(description="Prompt for outfit gen.", example="Make outfit for dating.")
    amount: int = Field(description="Prompt for outfit gen.", example=10, default=10)

    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    category: ImageCategory = Field(
        description="Image category.",
        example=ImageCategory.UPPER_BODY,
        default=ImageCategory.UPPER_BODY,
    )

class OutfitGenClothesCategory(BaseOutfitGenModel):
    clothes_id: UUID4 = OutfitGenFields.clothes_id
    category: ImageCategory = OutfitGenFields.category

class OutfitGenClothes(BaseOutfitGenModel):
    score_id: int = OutfitGenFields.amount
    clothes_id: UUID4 = OutfitGenFields.clothes_id

class OutfitGenTaskCmd(BaseOutfitGenModel):
    user_id: UUID4 = OutfitGenFields.user_id
    prompt: str = OutfitGenFields.prompt
    amount: int = OutfitGenFields.amount

    clothes: List[OutfitGenClothesCategory]

class OutfitGenResponseCmd(OutfitGenTaskCmd):
    user_id: UUID4 = OutfitGenFields.user_id
    prompt: str = OutfitGenFields.prompt
    amount: int = OutfitGenFields.amount

    clothes: List[OutfitGenClothes]
