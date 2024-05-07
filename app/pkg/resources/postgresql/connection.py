"""Create connection to postgresql."""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

import aiopg
from aiopg.pool import Cursor
from psycopg2.extras import RealDictCursor  # type: ignore

from app.pkg.settings import settings

__all__ = ["get_connection"]


@asynccontextmanager
async def get_connection(dsn=settings.RESOURCES.POSTGRES.DSN) -> AsyncGenerator[Cursor, None]:
    async with aiopg.connect(dsn=dsn) as conn:
        async with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
