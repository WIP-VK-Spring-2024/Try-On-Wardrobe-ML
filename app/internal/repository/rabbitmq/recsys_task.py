"""Repository for recsys model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import RecSysTaskCmd
from app.pkg.settings import settings

class RecSysTaskRepository(BaseRepository):
    """Repository for recsys model task queue."""

    QUEUE_NAME: str = settings.ML.RECSYS_TASK_QUEUE
    AVG_RESPONSE_TIME: int = 0.1

    async def create(
        self,
        cmd: RecSysTaskCmd,
    ) -> RecSysTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[RecSysTaskCmd, None]:
        async for message in self._read(model=RecSysTaskCmd):
            yield message
