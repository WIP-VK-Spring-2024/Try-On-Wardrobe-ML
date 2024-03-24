"""Models of try on model task object."""

import uuid

from pydantic import PositiveInt
from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings

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
    file_dir: str = Field(
        description="file dir path.",
        example="try_on/102",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )
    file_path: str = Field(
        description="file full path.",
        example="try_on/102/11.png",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )


class TryOnTaskCmd(BaseTryOnModel):
    user_id: UUID4 = TryOnFields.user_id
    clothes_id: UUID4 = TryOnFields.clothes_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    user_image_dir: str = TryOnFields.file_dir
    clothes_dir: str = TryOnFields.file_dir


class TryOnResponseCmd(BaseTryOnModel):
    user_id: UUID4 = TryOnFields.user_id
    clothes_id: UUID4 = TryOnFields.clothes_id
    user_image_id: UUID4 = TryOnFields.user_image_id
    try_on_result_path: str = TryOnFields.file_path
