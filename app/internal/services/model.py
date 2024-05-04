"""Model service."""
from typing import List

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
from app.internal.repository.rabbitmq.try_on_task import TryOnTaskRepository
from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.outfit_gen_task import OutfitGenTaskRepository
from app.internal.repository.rabbitmq.recsys_task import RecSysTaskRepository
from app.internal.repository.postgresql.clothes_vector import ClothesVectorRepository
from app.internal.repository.postgresql.outfit import OutfitRepository
from app.pkg.logger import get_logger

__all__ = ["ModelService"]


logger = get_logger(__name__)

class ModelService:

    cut_repository: CutTaskRepository
    try_on_repository: TryOnTaskRepository
    outfit_gen_repository: OutfitGenTaskRepository
    recsys_repository: RecSysTaskRepository
    clothes_vector_repository: ClothesVectorRepository
    outfit_repository: OutfitRepository

    def __init__(self) -> None:
        self.cut_repository: CutTaskRepository = CutTaskRepository()
        self.try_on_repository: TryOnTaskRepository = TryOnTaskRepository()
        self.outfit_gen_repository: OutfitGenTaskRepository = OutfitGenTaskRepository()
        self.recsys_repository: RecSysTaskRepository = RecSysTaskRepository()
        self.clothes_vector_repository: ClothesVectorRepository = ClothesVectorRepository()
        self.outfit_repository: OutfitRepository = OutfitRepository()


    async def create_try_on_task(self, cmd: TryOnTaskCmd) -> TryOnTaskCmd:
        logger.info("Got try on task request [%s]", cmd)
        return await self.try_on_repository.create(cmd=cmd)
    
    async def create_cut_task(self, cmd: CutTaskCmd) -> CutTaskCmd:
        logger.info("Got cut task request [%s]", cmd)
        return await self.cut_repository.create(cmd=cmd)
    
    async def create_outfit_gen_task(self, cmd: OutfitGenTaskCmd) -> OutfitGenTaskCmd:
        logger.info("Got outfit gen request [%s]", cmd)
        return await self.outfit_gen_repository.create(cmd=cmd)

    async def create_recsys_task(self, cmd: RecSysTaskCmd) -> RecSysTaskCmd:
        logger.info("Got recsys task request [%s]", cmd)
        return await self.recsys_repository.create(cmd=cmd)

    async def create_clothes_vector(self, cmd: ClothesVectorCreateCmd) -> ClothesVector:
        return await self.clothes_vector_repository.create(cmd=cmd)
    
    async def get_all_clothes_vector(self) -> List[ClothesVector]:
        return await self.clothes_vector_repository.read_all()

    async def get_all_outfit(self) -> List[Outfit]:
        return await self.outfit_repository.read_all()

    async def get_all_outfit_clothes_tensors(self) -> List[UserOutfitClothes]:
        return await self.outfit_repository.read_all_clothes_tensors()
