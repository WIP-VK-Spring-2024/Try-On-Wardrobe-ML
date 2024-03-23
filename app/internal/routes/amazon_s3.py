"""Amazon S3 router."""
from typing import Optional

from fastapi import File, UploadFile, Depends, status, APIRouter
from fastapi.responses import StreamingResponse

from app.internal.services import AmazonS3Service
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
    folder: Optional[str] = None,
    aws_service: AmazonS3Service = Depends(AmazonS3Service),
):
    aws_service.upload(file=file.file, file_name=file.filename, folder=folder)
    return ResponseMessage(message=f"File {folder}/{file.filename} successfully uploaded.")

@aws_router.delete(
    "/delete/{file_name}",
    response_model=ResponseMessage,
    status_code=status.HTTP_200_OK,
    description="Delete image from AWS.",
)
async def delete_file(
    file_name: str,
    folder: Optional[str] = None,
    aws_service: AmazonS3Service = Depends(AmazonS3Service),
):
    aws_service.delete(file_name=file_name, folder=folder)
    return ResponseMessage(message=f"File {folder}/{file_name} deleted successfully")


@aws_router.get(
    "/read/{file_name}",
    status_code=status.HTTP_200_OK,
    description="Get image from AWS.",
)
async def read(
    file_name: str,
    folder: Optional[str] = None,
    aws_service: AmazonS3Service = Depends(AmazonS3Service),
):
    file_content = aws_service.read(file_name=file_name, folder=folder)
    return StreamingResponse(
        file_content, 
        media_type='application/octet-stream',
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )
