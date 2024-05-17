"""Repository for cut model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CutResponseCmd
from app.pkg.settings import settings

class CutRespRepository(BaseRepository):
    """Repository for cut model response queue."""

    QUEUE_NAME: str = settings.ML.CUT_RESPONSE_QUEUE
    AVG_RESPONSE_TIME: int = 0.5

    async def create(
        self,
        cmd: CutResponseCmd,
    ) -> CutResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CutResponseCmd, None]:
        async for message in self._read(model=CutResponseCmd):
            yield message
