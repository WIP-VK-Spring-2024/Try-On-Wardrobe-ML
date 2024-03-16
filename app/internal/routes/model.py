import uuid
from pathlib import Path

from fastapi import File, UploadFile, status, Depends, APIRouter
from pydantic import UUID4
from app.pkg.logger import get_logger
from app.pkg.settings import settings
from app.pkg.models import ImageType, CreateTaskCmd, CreateTaskFileCmd, ResponseMessage
from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.services import FileService

logger = get_logger(__name__)

model_router = APIRouter(
    prefix="/data",
    tags=["Model"],
    responses={},
)

@model_router.post(
    "/load",
    response_model=ResponseMessage,
    status_code=status.HTTP_201_CREATED,
    description="Upload image.",
)
async def upload(
    user_id: UUID4,
    person_file: UploadFile = File(...),
    clothes_file: UploadFile = File(...),
    model_repository: ModelTaskRepository = Depends(ModelTaskRepository),
    file_service: FileService = Depends(FileService),
):
    logger.info(
        "Got POST image request person file name [%s], clothes file name [%s]", 
        person_file.filename,
        clothes_file.filename,
    )
    person_file_name, person_file_path = await file_service.save_file(person_file)
    clothes_file_name, clothes_file_path = await file_service.save_file(clothes_file)

    cmd = CreateTaskFileCmd(
        user_id=user_id,
        person_file_name=person_file_name,
        person_file_path=str(person_file_path),
        clothes_file_name=clothes_file_name,
        clothes_file_path=str(clothes_file_path),
    )

    await model_repository.create(cmd=cmd)
    
    return ResponseMessage(message=f"Successfully uploaded full-body")
