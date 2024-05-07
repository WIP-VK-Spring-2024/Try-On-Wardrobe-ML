"""Repository for outfit generation model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import OutfitGenTaskCmd
from app.pkg.settings import settings


class OutfitGenTaskRepository(BaseRepository):
    """Repository for outfit generation model task queue."""

    QUEUE_NAME: str = settings.ML.OUTFIT_GEN_TASK_QUEUE

    async def create(
        self,
        cmd: OutfitGenTaskCmd,
    ) -> OutfitGenTaskCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[OutfitGenTaskCmd, None]:
        async for message in self._read(model=OutfitGenTaskCmd):
            yield message
