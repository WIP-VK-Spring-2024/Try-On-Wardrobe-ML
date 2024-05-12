"""Base repository for rabbitmq."""
import json
from typing import AsyncGenerator

import aio_pika

from app.pkg.resources.rabbitmq.connection import get_connection
from app.internal.repository.rabbitmq.handlers.collect_response import (
    load_response,
)
from app.pkg.models.base import BaseModel
from app.pkg.models import RabbitMQInfo


class BaseRepository:
    """Base repository for rabbitmq."""

    QUEUE_NAME: str = "default"
    AVG_RESPONSE_TIME: int = 0

    async def get_queue_info(self) -> RabbitMQInfo:
        """Get the number of consumers listening to the queue."""
        avg_response_time = self.AVG_RESPONSE_TIME
        
        async with get_connection() as channel:
            try:
                queue = await channel.get_queue(self.QUEUE_NAME)
            except aio_pika.exceptions.ChannelNotFoundEntity as exc:
                return RabbitMQInfo(avg_response_time=avg_response_time)

        consumer_count = queue.declaration_result.consumer_count
        message_count = queue.declaration_result.message_count

        is_listening = consumer_count > 0
        response_time = avg_response_time * message_count

        res_model = RabbitMQInfo(
            consumer_count=consumer_count,
            message_count=message_count,
            avg_response_time=avg_response_time,
            is_listening=is_listening,
            response_time=response_time
        )
        return res_model

    async def _create(
        self,
        cmd: BaseModel,
    ) -> BaseModel:
        async with get_connection() as channel:
            await channel.declare_queue(self.QUEUE_NAME)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(cmd.to_dict(exclude_none=True)).encode(),
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
