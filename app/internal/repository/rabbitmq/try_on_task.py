"""Repository for try on model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import TryOnTaskCmd
from app.pkg.settings import settings

class TryOnTaskRepository(BaseRepository):
    """Repository for try on model task queue."""

    QUEUE_NAME: str = settings.ML.TRY_ON_TASK_QUEUE

    async def create(
        self,
        cmd: TryOnTaskCmd,
    ) -> TryOnTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[TryOnTaskCmd, None]:
        async for message in self._read(model=TryOnTaskCmd):
            yield message
