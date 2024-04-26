"""Models of status response object."""

from pydantic.fields import Field
from fastapi import status as fastapi_status

from app.pkg.models.base import BaseModel

__all__ = ["StatusResponse"]


class BaseStatusResponseModel(BaseModel):
    """Base model for status response model."""


class StatusResponseFields:
    """Model fields of status response model."""

    status_code: fastapi_status = Field(
        description="Image category.",
        example=fastapi_status.HTTP_404_NOT_FOUND,
        default=fastapi_status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    message: str = Field(
        description="Message for status.",
        example="Not found error",
        default="Internal server error",
    )

class StatusResponse(BaseStatusResponseModel):
    status_code: fastapi_status = StatusResponseFields.status_code
    message: str = StatusResponseFields.message
