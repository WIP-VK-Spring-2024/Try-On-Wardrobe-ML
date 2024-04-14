from fastapi import status, Depends, APIRouter
from app.pkg.models import TryOnTaskCmd, CutTaskCmd, OutfitGenTaskCmd
from app.internal.services.model import ModelService


model_router = APIRouter(
    prefix="/data",
    tags=["Model"],
    responses={},
)

@model_router.post(
    "/try_on",
    response_model=TryOnTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create try on task.",
)
async def try_on(
    cmd: TryOnTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.create_try_on_task(cmd=cmd)


@model_router.post(
    "/cut",
    response_model=CutTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create cut task.",
)
async def cut(
    cmd: CutTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.create_cut_task(cmd=cmd)


@model_router.post(
    "/outfit_gen",
    response_model=OutfitGenTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create rec sys task.",
)
async def cut(
    cmd: OutfitGenTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.create_outfit_gen_task(cmd=cmd)