"""Repository for model task queue."""
from typing import AsyncGenerator

from app.internal.repository.rabbitmq.base_repository import BaseRepository
from app.pkg.models import CreateTaskFileCmd


class ModelRepository(BaseRepository):
    """Repository for model task queue."""

    QUEUE_NAME: str = "model"

    async def create(
        self,
        cmd: CreateTaskFileCmd,
    ) -> CreateTaskFileCmd:
        return await self._create(cmd=cmd)

    async def read(self) -> AsyncGenerator[CreateTaskFileCmd, None]:
        async for message in self._read(model=CreateTaskFileCmd):
            yield message
