"""Cut service."""

from app.pkg.models import CutTaskCmd, RabbitMQInfo
from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.cut_response import CutRespRepository
from app.pkg.logger import get_logger

__all__ = ["CutService"]


logger = get_logger(__name__)

class CutService:

    task_repository: CutTaskRepository
    resp_repository: CutRespRepository

    def __init__(self) -> None:
        self.task_repository: CutTaskRepository = CutTaskRepository()
        self.resp_repository: CutRespRepository = CutRespRepository()

    async def create_cut_task(self, cmd: CutTaskCmd) -> CutTaskCmd:
        logger.info("Got cut task request [%s]", cmd)
        return await self.task_repository.create(cmd=cmd)

    async def get_resp_repository_info(self) -> RabbitMQInfo:
        return await self.resp_repository.get_queue_info()