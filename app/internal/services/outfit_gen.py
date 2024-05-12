"""Outfit gen service."""

from app.pkg.models import OutfitGenTaskCmd, RabbitMQInfo
from app.internal.repository.rabbitmq.outfit_gen_task import OutfitGenTaskRepository
from app.internal.repository.rabbitmq.outfit_gen_response import OutfitGenRespRepository
from app.pkg.logger import get_logger

__all__ = ["OutfitGenService"]


logger = get_logger(__name__)

class OutfitGenService:

    task_repository: OutfitGenTaskRepository
    resp_repository: OutfitGenRespRepository

    def __init__(self) -> None:
        self.task_repository: OutfitGenTaskRepository = OutfitGenTaskRepository()
        self.resp_repository: OutfitGenRespRepository = OutfitGenRespRepository()

    async def create_outfit_gen_task(self, cmd: OutfitGenTaskCmd) -> OutfitGenTaskCmd:
        logger.info("Got outfit gen request [%s]", cmd)
        return await self.task_repository.create(cmd=cmd)

    async def get_resp_repository_info(self) -> RabbitMQInfo:
        return await self.resp_repository.get_queue_info()