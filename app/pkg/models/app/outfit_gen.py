"""Models of outfit gen model."""

import uuid
from typing import List, Optional

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.models.app.image_category import ImageCategoryAutoset
from app.pkg.models.app.status_response import StatusResponse

__all__ = [
    "OutfitGenClothes",
    "OutfitGenTaskCmd",
    "OutfitGenResponseCmd",
    "Outfit",
    "OutfitGenClothesCategory",
]


class BaseOutfitGenModel(BaseModel):
    """Base model for outfit gen model."""


class OutfitGenFields:
    """Model fields of outfit gen model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    prompt: str = Field(description="Prompt for outfit gen.", example="Make outfit for dating.", default="")
    amount: int = Field(description="Prompt for outfit gen.", example=10, default=10)

    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    category: ImageCategoryAutoset = Field(
        description="Image category.",
        example=ImageCategoryAutoset.UPPER_BODY,
        default=ImageCategoryAutoset.UPPER_BODY,
    )

class OutfitGenClothesCategory(BaseOutfitGenModel):
    clothes_id: UUID4 = OutfitGenFields.clothes_id
    category: ImageCategoryAutoset = OutfitGenFields.category

class OutfitGenTaskCmd(BaseOutfitGenModel):
    user_id: UUID4 = OutfitGenFields.user_id
    prompt: str = OutfitGenFields.prompt
    amount: int = OutfitGenFields.amount

    clothes: List[OutfitGenClothesCategory]

class OutfitGenClothes(BaseOutfitGenModel):
    clothes_id: UUID4 = OutfitGenFields.clothes_id

class Outfit(BaseOutfitGenModel):
    clothes: List[OutfitGenClothes]

class OutfitGenResponseCmd(BaseOutfitGenModel, StatusResponse):
    user_id: UUID4 = OutfitGenFields.user_id
    outfits: Optional[List[Outfit]]
