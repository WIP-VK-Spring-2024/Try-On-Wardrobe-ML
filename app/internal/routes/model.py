import uuid
from pathlib import Path

from fastapi import File, UploadFile, status, Depends, APIRouter
from pydantic import UUID4
from app.pkg.logger import get_logger
from app.pkg.settings import settings
from app.pkg.models import ImageType, CreateTaskCmd, CreateTaskFileCmd, ResponseMessage
from app.internal.repository.rabbitmq.model_task import ModelTaskRepository


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
    image_type: ImageType,
    user_id: UUID4,
    image_id: UUID4,
    file: UploadFile = File(...),
    model_repository: ModelTaskRepository = Depends(ModelTaskRepository),
):
    cmd = CreateTaskCmd(
        image_type=image_type,
        user_id=user_id,
        image_id=image_id,
        file_name=file.filename,
    )
    logger.info("Got POST image request filename [%s]", file.filename)
    extension = file.filename.split(".")[-1]
    save_filename = f'{cmd.image_id}.{extension}'
    # save_filename = f'{uuid.uuid4()}.{extension}'

    default_path = str(settings.API_FILESYSTEM_FOLDER)
    if cmd.image_type == ImageType.FULL_BODY.value:
        file_path = f"{settings.API_FILESYSTEM_FOLDER}/try-on-full-body"

    elif cmd.image_type == ImageType.CLOTH.value:
        file_path = f"{settings.API_FILESYSTEM_FOLDER}/try-on-cloth"

    else:
        raise Exception(f"Incorrect input image_type. Should be one of: {ImageType.CLOTH}, {ImageType.FULL_BODY}")

    if not Path(file_path).exists():
        Path(file_path).mkdir(exist_ok=True, parents=True)

    file_path = Path(file_path, save_filename).absolute()

    cmd = CreateTaskFileCmd(
        image_type=cmd.image_type,
        user_id=cmd.user_id,
        image_id=cmd.image_id,
        file_name=save_filename,
        file_path=str(file_path),
    )

    contents = await file.read()
    with open(cmd.file_path, 'wb') as f:
        f.write(contents)

    await model_repository.create(cmd=cmd)
    
    return ResponseMessage(message=f"Successfully uploaded full-body")
