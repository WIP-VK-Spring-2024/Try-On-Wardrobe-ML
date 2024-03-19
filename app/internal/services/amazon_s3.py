"""Amazon S3 service."""

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile

from app.pkg.models.base.exception import BaseAPIException
from app.pkg.settings import settings
from app.pkg.logger import get_logger

__all__ = ["AmazonS3Service"]

logger = get_logger(__name__)

class AmazonS3Service:

    def __init__(self) -> None:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS.ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS.SECRET_ACCESS_KEY,
            region_name=settings.AWS.REGION_NAME
        )
        self.s3_bucket_name = settings.AWS.BUCKET_NAME


    def upload(self, file: UploadFile = File(...)):
        try:
            res = self.s3.upload_fileobj(file.file, self.s3_bucket_name, file.filename)
            logger.info("Uploaded file to S3, response: %s", res)
        except ClientError as e:
            raise BaseAPIException(message=f"Failed to upload file to S3: {e}")
        
    def delete_file(self, file_name: str):
        try:
            res = self.s3.delete_object(Bucket=self.s3_bucket_name, Key=file_name)
            logger.info("Deleted file from S3, response: %s", res)
        except ClientError as e:
            raise BaseAPIException(message=f"Failed to delete file from S3: {e}")
