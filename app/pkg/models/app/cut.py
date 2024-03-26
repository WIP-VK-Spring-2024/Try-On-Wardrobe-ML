"""Models of cut model task object."""

import uuid

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings

__all__ = [
    "CutTaskCmd",
    "CutResponseCmd",
]


class BaseCutModel(BaseModel):
    """Base model cut model."""


class CutFields:
    """Model fields of cut model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    clothes_id: UUID4 = Field(description="Clothes id.", example=uuid.uuid4())
    file_dir: str = Field(
        description="file dir path.",
        example="try_on/102",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )

class CutTaskCmd(BaseCutModel):
    user_id: UUID4 = CutFields.user_id
    clothes_id: UUID4 = CutFields.clothes_id
    clothes_dir: str = CutFields.file_dir


class CutResponseCmd(BaseCutModel):
    user_id: UUID4 = CutFields.user_id
    clothes_id: UUID4 = CutFields.clothes_id
    result_dir: str = CutFields.file_dir
