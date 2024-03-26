"""Repository for cut model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CutResponseCmd


class CutRespRepository(BaseRepository):
    """Repository for cut model response queue."""

    QUEUE_NAME: str = "cut_response"

    async def create(
        self,
        cmd: CutResponseCmd,
    ) -> CutResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CutResponseCmd, None]:
        async for message in self._read(model=CutResponseCmd):
            yield message
