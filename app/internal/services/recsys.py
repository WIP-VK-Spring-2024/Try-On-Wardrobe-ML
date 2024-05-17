"""Recsys service."""

from app.pkg.models import RecSysTaskCmd, RabbitMQInfo
from app.internal.repository.rabbitmq.recsys_task import RecSysTaskRepository
from app.internal.repository.rabbitmq.recsys_response import RecSysRespRepository
from app.pkg.logger import get_logger

__all__ = ["RecSysService"]


logger = get_logger(__name__)

class RecSysService:

    task_repository: RecSysTaskRepository
    resp_repository: RecSysRespRepository

    def __init__(self) -> None:
        self.task_repository: RecSysTaskRepository = RecSysTaskRepository()
        self.resp_repository: RecSysRespRepository = RecSysRespRepository()

    async def create_recsys_task(self, cmd: RecSysTaskCmd) -> RecSysTaskCmd:
        logger.info("Got recsys task request [%s]", cmd)
        return await self.task_repository.create(cmd=cmd)

    async def get_resp_repository_info(self) -> RabbitMQInfo:
        return await self.resp_repository.get_queue_info()
    
    async def get_task_repository_info(self) -> RabbitMQInfo:
        return await self.task_repository.get_queue_info()
