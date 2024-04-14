"""Models of cut model task object."""

import uuid
from typing import List, Dict

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.settings import settings

__all__ = [
    "CutTaskCmd",
    "ClothesTaskClassification",
    "CutResponseCmd",
    "ClothesRespClassification",
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

    probability: Dict[str, float] = Field(
        description="Classification probability.",
        example="dress",
    )
    seasons: List[str] = Field(
        description="Image seasons.",
        example=["winter"],
    )
    categories: List[str] = Field(
        description="Image categories.",
        example=["dress"],
    )
    subcategories: List[str] = Field(
        description="Image subcategories.",
        example=["jacket"],
    )
    tags: List[str] = Field(
        description="Image tags.",
        example=["outfit"],
        default=[],
    )
    styles: List[str] = Field(
        description="Image tag.",
        example=["school"],
    )

class ClothesTaskClassification(BaseCutModel):
    seasons: List[str] = CutFields.seasons
    categories: List[str] = CutFields.categories
    subcategories: List[str] = CutFields.subcategories
    tags: List[str] = CutFields.tags
    styles: List[str] = CutFields.styles

class CutTaskCmd(BaseCutModel):
    user_id: UUID4 = CutFields.user_id
    clothes_id: UUID4 = CutFields.clothes_id
    clothes_dir: str = CutFields.file_dir
    classification: ClothesTaskClassification

class ClothesRespClassification(BaseCutModel):
    seasons: Dict[str, float] = CutFields.probability
    categories: Dict[str, float] = CutFields.probability
    subcategories: Dict[str, float] = CutFields.probability
    tags: Dict[str, float] = CutFields.probability
    styles: Dict[str, float] = CutFields.probability

class CutResponseCmd(CutTaskCmd):
    user_id: UUID4 = CutFields.user_id
    clothes_id: UUID4 = CutFields.clothes_id
    clothes_dir: str = CutFields.file_dir
    classification: ClothesRespClassification
