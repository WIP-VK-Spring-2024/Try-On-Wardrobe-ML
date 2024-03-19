"""Model service."""
import uuid
from pathlib import Path

from pydantic import UUID4
from fastapi import File, UploadFile, status, Depends, APIRouter

from app.pkg.settings import settings
from app.pkg.models import ImageType, CreateTaskCmd, CreateTaskFileCmd, ResponseMessage
from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.services import FileService
from app.pkg.logger import get_logger

__all__ = ["ModelService"]


logger = get_logger(__name__)

class ModelService:

    model_repository: ModelTaskRepository
    file_service: FileService

    def __init__(self) -> None:
        self.model_repository: ModelTaskRepository = ModelTaskRepository()
        self.file_service: FileService = FileService()

    async def put_in_queue(
        self,
        clothes_id: UUID4,
        user_id: UUID4,
        person_file: UploadFile = File(...),
        clothes_file: UploadFile = File(...),
    ):
        logger.info(
            "Got POST image request person file name [%s], clothes file name [%s]", 
            person_file.filename,
            clothes_file.filename,
        )
        person_file_name, person_file_path = await self.file_service.save_file(person_file)
        clothes_file_name, clothes_file_path = await self.file_service.save_file(clothes_file)

        cmd = CreateTaskFileCmd(
            clothes_id=clothes_id,
            user_id=user_id,
            person_file_name=person_file_name,
            person_file_path=str(person_file_path),
            clothes_file_name=clothes_file_name,
            clothes_file_path=str(clothes_file_path),
        )

        await self.model_repository.create(cmd=cmd)