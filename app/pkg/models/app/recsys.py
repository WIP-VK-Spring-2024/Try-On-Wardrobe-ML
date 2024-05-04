"""Models of recsys model task object."""

import uuid
from typing import List, Optional

from pydantic.fields import Field
from pydantic import UUID4

from app.pkg.models.base import BaseModel
from app.pkg.models.app.status_response import StatusResponse

__all__ = [
    "RecSysTaskCmd",
    "RecSysResponseCmd",
]


class BaseRecSysModel(BaseModel):
    """Base model for recsys model."""


class RecsysFields:
    """Model fields of recsys model."""

    user_id: UUID4 = Field(description="User id.", example=uuid.uuid4())
    samples_amount: int = Field(description="Recsys sample amount.", default=5)

    outfits_ids: List[UUID4] = Field(
        description="Outfits ids for recsys result.",
        default_factory=lambda: [],
    )

class RecSysTaskCmd(BaseRecSysModel):
    user_id: UUID4 = RecsysFields.user_id
    samples_amount: int = RecsysFields.samples_amount


class RecSysResponseCmd(BaseRecSysModel, StatusResponse):
    outfits_ids: List[UUID4] = RecsysFields.outfits_ids
