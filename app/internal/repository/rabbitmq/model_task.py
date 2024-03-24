"""Repository for model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import TryOnTaskCmd


class ModelTaskRepository(BaseRepository):
    """Repository for model task queue."""

    QUEUE_NAME: str = "try_on"

    async def create(
        self,
        cmd: TryOnTaskCmd,
    ) -> TryOnTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[TryOnTaskCmd, None]:
        async for message in self._read(model=TryOnTaskCmd):
            yield message
