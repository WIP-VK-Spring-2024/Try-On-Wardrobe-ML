from fastapi import File, UploadFile, status, Depends, APIRouter
from pydantic import UUID4
from app.pkg.settings import settings
from app.pkg.models import ResponseMessage
from app.internal.services.model import ModelService


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
    clothes_id: UUID4,
    user_id: UUID4,
    person_file: UploadFile = File(...),
    clothes_file: UploadFile = File(...),
    model_service: ModelService = Depends(ModelService),
):
    await model_service.put_in_queue(clothes_id, user_id, person_file, clothes_file)
    return ResponseMessage(message=f"Successfully uploaded full-body")
