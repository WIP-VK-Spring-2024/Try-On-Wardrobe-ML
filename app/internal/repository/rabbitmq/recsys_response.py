"""Repository for recsys model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import RecSysResponseCmd


class RecSysRespRepository(BaseRepository):
    """Repository for recsys model response queue."""

    QUEUE_NAME: str = "recsys_response"

    async def create(
        self,
        cmd: RecSysResponseCmd,
    ) -> RecSysResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[RecSysResponseCmd, None]:
        async for message in self._read(model=RecSysResponseCmd):
            yield message
