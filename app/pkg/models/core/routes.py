"""Model for contains ``APIRouter`` instance."""

from dataclasses import dataclass
from typing import Tuple

from fastapi import APIRouter, FastAPI

__all__ = ["Routes"]


@dataclass(frozen=True)
class Routes:
    """Frozen model for storage all ``APIRouter``.

    Attributes:
        routers:
            Tuple of ``APIRouter`` instances.
    """

    routers: Tuple[APIRouter, ...]

    def register_routes(self, app: FastAPI):
        for router in self.routers:
            app.include_router(router)
