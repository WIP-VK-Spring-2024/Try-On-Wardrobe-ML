"""Base repository for rabbitmq."""
import json
from typing import AsyncGenerator

import aio_pika

from app.pkg.resources.rabbitmq.connection import get_connection
from app.internal.repository.rabbitmq.handlers.collect_response import (
    load_response,
)
from app.pkg.models.base import BaseModel


class BaseRepository:
    """Base repository for rabbitmq."""

    QUEUE_NAME: str = "default"

    async def _create(
        self,
        cmd: BaseModel,
    ) -> BaseModel:
        async with get_connection() as channel:
            await channel.declare_queue(self.QUEUE_NAME)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(cmd.to_dict()).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                self.QUEUE_NAME,
            )

            return cmd

    async def _read(
        self,
        model: BaseModel,
    ) -> AsyncGenerator[BaseModel, None]:
        async with get_connection() as channel:
            queue = await channel.declare_queue(self.QUEUE_NAME)
            async for message in queue.iterator():
                await message.ack()
                decoded = await load_response(message, model)
                yield decoded

    async def create(self, cmd: BaseModel) -> BaseModel:
        raise NotImplementedError

    async def read(self) -> AsyncGenerator[BaseModel, None]:
        raise NotImplementedError
