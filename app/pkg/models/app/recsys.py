"""Models of recsys model."""

import uuid
from typing import List

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings
from app.pkg.models.app.image_category import ImageCategory

__all__ = [
    "RecSysClothes",
    "RecSysTaskCmd",
    "RecSysResponseCmd",
]


class BaseRecSysModel(BaseModel):
    """Base model for recsys model."""


class RecSysFields:
    """Model fields of try on model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    prompt: str = Field(description="Prompt for recsys.", example="Make outfit for dating.")

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

class RecSysClothes(BaseRecSysModel):
    clothes_id: UUID4 = RecSysFields.clothes_id
    category: ImageCategory = RecSysFields.category


class RecSysTaskCmd(BaseRecSysModel):
    user_id: UUID4 = RecSysFields.user_id
    clothes_id: UUID4 = RecSysFields.clothes_id

    clothes: List[RecSysClothes]


class RecSysResponseCmd(RecSysTaskCmd):
    pass
