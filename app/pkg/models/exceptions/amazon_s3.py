from typing import Optional

from fastapi import status

from app.pkg.models.base import BaseAPIException

__all__ = [
    "AmazonS3NotFoundError",
    "AmazonS3UploadError",
]


class AmazonS3Error(BaseAPIException):
    """Exception for amazon s3 errors."""

    message = "Amazon S3 internal error."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

class AmazonS3NotFoundError(AmazonS3Error):
    message = "Failed to read file from S3."
    status_code = status.HTTP_404_NOT_FOUND

class AmazonS3UploadError(AmazonS3Error):
    message = "Failed to upload file to S3."
    status_code = status.HTTP_406_NOT_ACCEPTABLE
