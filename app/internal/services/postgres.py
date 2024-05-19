"""Postgres service."""
from typing import List

from app.pkg.models import (
    ClothesVectorCreateCmd,
    ClothesVector,
    Outfit,
    UserOutfitClothes,
)
from app.internal.repository.postgresql.clothes_vector import ClothesVectorRepository
from app.internal.repository.postgresql.outfit import OutfitRepository
from app.pkg.logger import get_logger

__all__ = ["PostgresService"]


logger = get_logger(__name__)

class PostgresService:

    clothes_vector_repository: ClothesVectorRepository
    outfit_repository: OutfitRepository

    def __init__(self) -> None:
        self.clothes_vector_repository: ClothesVectorRepository = ClothesVectorRepository()
        self.outfit_repository: OutfitRepository = OutfitRepository()

    async def create_clothes_vector(self, cmd: ClothesVectorCreateCmd) -> ClothesVector:
        return await self.clothes_vector_repository.create(cmd=cmd)
    
    async def get_all_clothes_vector(self) -> List[ClothesVector]:
        return await self.clothes_vector_repository.read_all()

    async def get_all_outfit(self) -> List[Outfit]:
        return await self.outfit_repository.read_all()

    async def get_all_outfit_clothes_tensors(self) -> List[UserOutfitClothes]:
        return await self.outfit_repository.read_all_clothes_tensors()
