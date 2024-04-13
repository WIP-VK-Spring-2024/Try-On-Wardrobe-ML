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


class BaseRecSysModel(BaseModel):
    """Base model for recsys model."""


class OutfitGenFields:
    """Model fields of try on model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    prompt: str = Field(description="Prompt for recsys.", example="Make outfit for dating.")

    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    category: ImageCategory = Field(
        description="Image category.",
        example=ImageCategory.UPPER_BODY,
        default=ImageCategory.UPPER_BODY,
    )

class OutfitGenClothes(BaseRecSysModel):
    clothes_id: UUID4 = OutfitGenFields.clothes_id
    category: ImageCategory = OutfitGenFields.category


# class RecSysForward(BaseRecSysModel):
#     clothes_id: UUID4 = OutfitGenFields.clothes_id
#     clothes_image: bytes

class OutfitGenTaskCmd(BaseRecSysModel):
    user_id: UUID4 = OutfitGenFields.user_id
    prompt: str = OutfitGenFields.prompt

    clothes: List[OutfitGenClothes]


class OutfitGenResponseCmd(OutfitGenTaskCmd):
    pass
