"""Repository for model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import TryOnResponseCmd


class ModelRespRepository(BaseRepository):
    """Repository for model response queue."""

    QUEUE_NAME: str = "try_on_response"

    async def create(
        self,
        cmd: TryOnResponseCmd,
    ) -> TryOnResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[TryOnResponseCmd, None]:
        async for message in self._read(model=TryOnResponseCmd):
            yield message
