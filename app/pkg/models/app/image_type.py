"""Enum of Image Type object."""

from app.pkg.models.base import BaseEnum


__all__ = ["ImageType"]


class ImageType(BaseEnum):
    FULL_BODY: str = "full-body"
    CLOTH: str = "cloth"
