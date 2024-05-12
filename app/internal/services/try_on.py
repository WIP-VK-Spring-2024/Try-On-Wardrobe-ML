"""Try on service."""

from app.pkg.models import TryOnTaskCmd, RabbitMQInfo
from app.internal.repository.rabbitmq.try_on_task import TryOnTaskRepository
from app.internal.repository.rabbitmq.try_on_response import TryOnRespRepository
from app.pkg.logger import get_logger

__all__ = ["TryOnService"]


logger = get_logger(__name__)

class TryOnService:

    task_repository: TryOnTaskRepository
    resp_repository: TryOnTaskRepository

    def __init__(self) -> None:
        self.task_repository: TryOnTaskRepository = TryOnTaskRepository()
        self.resp_repository: TryOnRespRepository = TryOnRespRepository()

    async def create_try_on_task(self, cmd: TryOnTaskCmd) -> TryOnTaskCmd:
        logger.info("Got try on task request [%s]", cmd)
        return await self.task_repository.create(cmd=cmd)

    async def get_resp_repository_info(self) -> RabbitMQInfo:
        return await self.resp_repository.get_queue_info()
