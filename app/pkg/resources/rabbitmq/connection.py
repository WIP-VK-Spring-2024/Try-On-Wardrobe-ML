"""Create connection to rabbitmq."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import aio_pika

from app.pkg.settings import settings

__all__ = ["get_connection"]


@asynccontextmanager
async def get_connection(dsn = settings.RESOURCES.RABBITMQ.DSN) -> AsyncGenerator[aio_pika.Channel, None]:
    async with await aio_pika.connect_robust(dsn) as conn:
        channel = await conn.channel()
        yield channel
