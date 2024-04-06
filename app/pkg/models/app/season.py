"""Enum of Season object."""

from app.pkg.models.base import BaseEnum


__all__ = ["Season"]


class Season(BaseEnum):
    SUMMER: str = "summer"
    WINTER: str = "winter"
    AUTUMN: str = "autumn"
    SPRING: str = "spring"
