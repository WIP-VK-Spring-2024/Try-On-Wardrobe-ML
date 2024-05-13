"""Repository for cut model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CutTaskCmd
from app.pkg.settings import settings


class CutTaskRepository(BaseRepository):
    """Repository for cut model task queue."""

    QUEUE_NAME: str = settings.ML.CUT_TASK_QUEUE
    AVG_RESPONSE_TIME: int = 0.5

    async def create(
        self,
        cmd: CutTaskCmd,
    ) -> CutTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CutTaskCmd, None]:
        async for message in self._read(model=CutTaskCmd):
            yield message
