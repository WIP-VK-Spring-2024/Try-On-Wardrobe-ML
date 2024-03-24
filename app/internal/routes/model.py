from fastapi import status, Depends, APIRouter
from app.pkg.models import TryOnTaskCmd
from app.internal.services.model import ModelService


model_router = APIRouter(
    prefix="/data",
    tags=["Model"],
    responses={},
)

@model_router.post(
    "/load",
    response_model=TryOnTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Upload image.",
)
async def upload(
    cmd: TryOnTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.put_in_queue(cmd=cmd)
