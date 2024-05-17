from typing import List

from fastapi import status, Depends, APIRouter
from pydantic import UUID4

from app.pkg.models import (
    ClothesVectorCreateCmd,
    ClothesVector,
    Outfit,
    UserOutfitClothes,
)
from app.internal.services.postgres import PostgresService


postgres_router = APIRouter(
    prefix="/postgres",
    tags=["Postgres"],
    responses={},
)

@postgres_router.post(
    "/clothes_vector",
    response_model=ClothesVector,
    status_code=status.HTTP_201_CREATED,
    description="Create clothes vector.",
)
async def create_clothes_vector(
    clothes_id: UUID4,
    tensor: List[float],
    postgres_service: PostgresService = Depends(PostgresService),
):
    cmd = ClothesVectorCreateCmd(clothes_id=clothes_id, tensor=tensor)
    return await postgres_service.create_clothes_vector(cmd=cmd)

@postgres_router.get(
    "/clothes_vector",
    response_model=List[ClothesVector],
    status_code=status.HTTP_200_OK,
    description="Get all clothes vector.",
)
async def get_all_clothes_vector(
    postgres_service: PostgresService = Depends(PostgresService),
):
    return await postgres_service.get_all_clothes_vector()


@postgres_router.get(
    "/outfit",
    response_model=List[Outfit],
    status_code=status.HTTP_200_OK,
    description="Get all outfit.",
)
async def get_all_outfit(
    postgres_service: PostgresService = Depends(PostgresService),
):
    return await postgres_service.get_all_outfit()


@postgres_router.get(
    "/outfit_clothes_tensors",
    response_model=List[UserOutfitClothes],
    status_code=status.HTTP_200_OK,
    description="Get all outfit clothes tensors.",
)
async def get_all_outfit_clothes_tensors(
    postgres_service: PostgresService = Depends(PostgresService),
):
    return await postgres_service.get_all_outfit_clothes_tensors()
