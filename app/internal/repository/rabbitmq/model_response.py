"""Repository for model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CreateRespFileCmd


class ModelRespRepository(BaseRepository):
    """Repository for model task queue."""

    QUEUE_NAME: str = "model_response"

    async def create(
        self,
        cmd: CreateRespFileCmd,
    ) -> CreateRespFileCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CreateRespFileCmd, None]:
        async for message in self._read(model=CreateRespFileCmd):
            yield message
