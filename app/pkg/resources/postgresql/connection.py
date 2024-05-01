"""Create connection to postgresql."""

from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.pkg.settings import settings

__all__ = ["get_async_session"]


engine = create_async_engine(settings.RESOURCES.POSTGRES.DSN)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
