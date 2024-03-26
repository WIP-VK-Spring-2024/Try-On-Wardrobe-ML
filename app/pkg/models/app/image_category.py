"""Enum of Image Category object."""

from app.pkg.models.base import BaseEnum


__all__ = ["ImageCategory"]


class ImageCategory(BaseEnum):
    LOWER_BODY: str = "lower_body"
    DRESSES: str = "dresses"
    UPPER_BODY: str = "upper_body"
