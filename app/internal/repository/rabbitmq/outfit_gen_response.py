"""Repository for outfit generation model response queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import OutfitGenResponseCmd


class OutfitGenRespRepository(BaseRepository):
    """Repository for outfit generation model response queue."""

    QUEUE_NAME: str = "outfit_gen_response"

    async def create(
        self,
        cmd: OutfitGenResponseCmd,
    ) -> OutfitGenResponseCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[OutfitGenResponseCmd, None]:
        async for message in self._read(model=OutfitGenResponseCmd):
            yield message
