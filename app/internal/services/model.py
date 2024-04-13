"""Model service."""
from app.pkg.models import TryOnTaskCmd, CutTaskCmd, OutfitGenTaskCmd
from app.internal.repository.rabbitmq.try_on_task import TryOnTaskRepository
from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.outfit_gen_task import OutfitGenTaskRepository
from app.pkg.logger import get_logger

__all__ = ["ModelService"]


logger = get_logger(__name__)

class ModelService:

    cut_repository: CutTaskRepository
    try_on_repository: TryOnTaskRepository
    outfit_gen_repository: OutfitGenTaskRepository

    def __init__(self) -> None:
        self.cut_repository: CutTaskRepository = CutTaskRepository()
        self.try_on_repository: TryOnTaskRepository = TryOnTaskRepository()
        self.outfit_gen_repository: OutfitGenTaskRepository = OutfitGenTaskRepository()

    async def create_try_on_task(self, cmd: TryOnTaskCmd) -> TryOnTaskCmd:
        logger.info("Got image request [%s]", cmd)
        return await self.try_on_repository.create(cmd=cmd)
    
    async def create_cut_task(self, cmd: CutTaskCmd) -> CutTaskCmd:
        logger.info("Got image request [%s]", cmd)
        return await self.cut_repository.create(cmd=cmd)
    
    async def create_outfit_gen_task(self, cmd: OutfitGenTaskCmd) -> OutfitGenTaskCmd:
        logger.info("Got image request [%s]", cmd)
        return await self.outfit_gen_repository.create(cmd=cmd)
