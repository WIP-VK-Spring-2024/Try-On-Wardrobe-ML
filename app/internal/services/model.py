"""Model service."""
from app.pkg.models import TryOnTaskCmd
from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.pkg.logger import get_logger

__all__ = ["ModelService"]


logger = get_logger(__name__)

class ModelService:

    model_repository: ModelTaskRepository

    def __init__(self) -> None:
        self.model_repository: ModelTaskRepository = ModelTaskRepository()

    async def put_in_queue(self, cmd: TryOnTaskCmd) -> TryOnTaskCmd:
        logger.info("Got image request [%s]", cmd)
        return await self.model_repository.create(cmd=cmd)