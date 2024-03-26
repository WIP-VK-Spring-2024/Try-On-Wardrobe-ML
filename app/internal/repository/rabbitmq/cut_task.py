"""Repository for cut model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CutTaskCmd


class CutTaskRepository(BaseRepository):
    """Repository for cut model response queue."""

    QUEUE_NAME: str = "cut"

    async def create(
        self,
        cmd: CutTaskCmd,
    ) -> CutTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CutTaskCmd, None]:
        async for message in self._read(model=CutTaskCmd):
            yield message
