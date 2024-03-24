"""Models of try on model task object."""

import uuid

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings
from app.pkg.models.app.image_category import ImageCategory

__all__ = [
    "TryOnTaskCmd",
    "TryOnResponseCmd",
]


class BaseTryOnModel(BaseModel):
    """Base model for try on model."""


class TryOnFields:
    """Model fields of try on model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    user_image_id: UUID4 = Field(description="User photo id.", example=uuid.uuid4())
    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    category: ImageCategory = Field(
        description="Image category.",
        example=ImageCategory.UPPER_BODY,
        default=ImageCategory.UPPER_BODY,
    )
    file_dir: str = Field(
        description="file dir path.",
        example="try_on/102",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )

class TryOnTaskCmd(BaseTryOnModel):
    user_id: UUID4 = TryOnFields.user_id
    clothes_id: UUID4 = TryOnFields.clothes_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    user_image_dir: str = TryOnFields.file_dir
    clothes_dir: str = TryOnFields.file_dir
    category: ImageCategory = TryOnFields.category


class TryOnResponseCmd(BaseTryOnModel):
    user_id: UUID4 = TryOnFields.user_id
    clothes_id: UUID4 = TryOnFields.clothes_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    try_on_result_id: str = TryOnFields.user_image_id
    try_on_result_dir: str = TryOnFields.file_dir
