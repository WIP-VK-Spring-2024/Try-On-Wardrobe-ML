"""Repository for try on model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import TryOnResponseCmd
from app.pkg.settings import settings

class TryOnRespRepository(BaseRepository):
    """Repository for try on model response queue."""

    QUEUE_NAME: str = settings.ML.TRY_ON_RESPONSE_QUEUE
    AVG_RESPONSE_TIME: int = 9.4

    async def create(
        self,
        cmd: TryOnResponseCmd,
    ) -> TryOnResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[TryOnResponseCmd, None]:
        async for message in self._read(model=TryOnResponseCmd):
            yield message
