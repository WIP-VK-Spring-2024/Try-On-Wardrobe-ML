"""Amazon S3 router."""

from fastapi import File, UploadFile, Depends, status, APIRouter

from app.internal.services.amazon_s3 import AmazonS3Service
from app.pkg.models import ResponseMessage

aws_router = APIRouter(
    prefix="/aws",
    tags=["Amazon_S3"],
    responses={},
)

@aws_router.post(
    "/upload/",
    response_model=ResponseMessage,
    status_code=status.HTTP_201_CREATED,
    description="Upload image to AWS.",
)
async def upload_file(
    file: UploadFile = File(...),
    aws_service: AmazonS3Service = Depends(AmazonS3Service),
):
    aws_service.upload(file)
    return ResponseMessage(message="File successfully uploaded.")

@aws_router.delete(
    "/delete/{file_name}",
    response_model=ResponseMessage,
    status_code=status.HTTP_200_OK,
    description="Delete image from AWS.",
)
async def delete_file(
    file_name: str,
    aws_service: AmazonS3Service = Depends(AmazonS3Service),
):
    aws_service.delete_file(file_name=file_name)
    return ResponseMessage(message=f"File {file_name} deleted successfully")
