from typing import List

from fastapi import status, Depends, APIRouter
from pydantic import UUID4

from app.pkg.models import (
    TryOnTaskCmd,
    CutTaskCmd,
    OutfitGenTaskCmd,
    RecSysTaskCmd,
    ClothesVectorCreateCmd,
    ClothesVector,
    Outfit,
    UserOutfitClothes,
)
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
    description="Create outfit gen task.",
)
async def cut(
    cmd: OutfitGenTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.create_outfit_gen_task(cmd=cmd)

@model_router.post(
    "/recsys",
    response_model=RecSysTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create rec sys task.",
)
async def recsys(
    cmd: RecSysTaskCmd,
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.create_recsys_task(cmd=cmd)

@model_router.post(
    "/clothes_vector",
    response_model=ClothesVector,
    status_code=status.HTTP_201_CREATED,
    description="Create clothes vector.",
)
async def create_clothes_vector(
    clothes_id: UUID4,
    tensor: List[float],
    model_service: ModelService = Depends(ModelService),
):
    cmd = ClothesVectorCreateCmd(clothes_id=clothes_id, tensor=tensor)
    return await model_service.create_clothes_vector(cmd=cmd)

@model_router.get(
    "/clothes_vector",
    response_model=List[ClothesVector],
    status_code=status.HTTP_200_OK,
    description="Get all clothes vector.",
)
async def get_all_clothes_vector(
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.get_all_clothes_vector()


@model_router.get(
    "/outfit",
    response_model=List[Outfit],
    status_code=status.HTTP_200_OK,
    description="Get all outfit.",
)
async def get_all_outfit(
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.get_all_outfit()


@model_router.get(
    "/outfit_clothes_tensors",
    response_model=List[UserOutfitClothes],
    status_code=status.HTTP_200_OK,
    description="Get all outfit clothes tensors.",
)
async def get_all_outfit_clothes_tensors(
    model_service: ModelService = Depends(ModelService),
):
    return await model_service.get_all_outfit_clothes_tensors()
