"""Models of try on model task object."""

import uuid
from typing import List, Optional

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings
from app.pkg.models.app.image_category import ImageCategory
from app.pkg.models.app.status_response import StatusResponse

__all__ = [
    "TryOnClothes",
    "TryOnImageClothes",
    "TryOnTaskCmd",
    "TryOnResponseCmd",
]


class BaseTryOnModel(BaseModel):
    """Base model for try on model."""


class TryOnFields:
    """Model fields of try on model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    outfit_id: Optional[UUID4] = Field(description="Outfit id.", example=uuid.uuid4())
    user_image_id: UUID4 = Field(description="User photo id.", example=uuid.uuid4())
    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    category: ImageCategory = Field(
        description="Image category.",
        example=ImageCategory.UPPER_BODY,
        default=ImageCategory.UPPER_BODY,
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Image subcategory for model description.",
    )
    file_dir: str = Field(
        description="file dir path.",
        example="try_on/102",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )

class TryOnClothes(BaseTryOnModel):
    clothes_id: UUID4 = TryOnFields.clothes_id
    category: ImageCategory = TryOnFields.category
    subcategory: Optional[str] = TryOnFields.subcategory

class TryOnListFields:
    """Model list fields of try on model."""

    clothes: List[TryOnClothes] = Field(description="List of clothes")

class TryOnTaskCmd(BaseTryOnModel):
    user_id: UUID4 = TryOnFields.user_id
    outfit_id: Optional[UUID4] = TryOnFields.outfit_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    user_image_dir: str = TryOnFields.file_dir
    
    clothes: List[TryOnClothes] = TryOnListFields.clothes


class TryOnResponseCmd(BaseTryOnModel, StatusResponse):
    user_id: UUID4 = TryOnFields.user_id
    outfit_id: Optional[UUID4] = TryOnFields.outfit_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    clothes: List[TryOnClothes] = TryOnListFields.clothes

    # TODO: Move dirs to .env
    try_on_dir: Optional[str] = TryOnFields.file_dir
    try_on_id: Optional[UUID4] = TryOnFields.clothes_id
