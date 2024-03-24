"""Models of amazon s3 object."""

from pydantic.fields import Field

from app.pkg.models.base import BaseModel

__all__ = ["ResponseMessage"]


class BaseAmazonS3(BaseModel):
    """Base model for amazon s3."""


class AmazonS3Fields:
    """Model fields of amazon s3."""

    message: str = Field(description="Message.", example="Successfully uploaded.")


class ResponseMessage(BaseAmazonS3):
    message: str = AmazonS3Fields.message