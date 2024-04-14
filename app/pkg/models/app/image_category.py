"""Enum of Image Category object."""

from app.pkg.models.base import BaseEnum


__all__ = [
    "ImageCategory",
    "ImageCategoryAutoset",
]


class ImageCategory(BaseEnum):
    LOWER_BODY: str = "lower_body"
    DRESSES: str = "dresses"
    UPPER_BODY: str = "upper_body"

class ImageCategoryAutoset(BaseEnum):
    LOWER_BODY: str = "lower garment"
    DRESSES: str = "dress"
    UPPER_BODY: str = "upper garment"
    OUTWEAR: str = "outerwear"
