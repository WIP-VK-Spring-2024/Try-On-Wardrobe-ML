"""Repository for test."""

from typing import List

from sqlalchemy import insert, select

from app.internal.repository.repository import Repository
from app.pkg.resources.postgresql.connection import get_async_session
from app.pkg.models.postgresql.clothes_vector import ClothesVector
from app.internal.repository.postgresql.handlers.handle_exception import handle_exception
from app.pkg import models

__all__ = ["ClothesVectorRepository"]


class ClothesVectorRepository(Repository):
    """Clothes vector repository implementation."""

    @handle_exception
    async def create(self, cmd: models.ClothesVectorCreateCmd) -> models.ClothesVector:
        command = (
            insert(ClothesVector)
            .values(**cmd.dict())
            .returning(ClothesVector)
        )

        async with get_async_session() as session:
            result = await session.execute(command)
            await session.commit()
            orm_result = result.scalars().first()
    
        return models.ClothesVector.from_orm(orm_result)

    @handle_exception
    async def read_all(self) -> List[models.ClothesVector]:
        query = select(ClothesVector)

        async with get_async_session() as session:
            result = await session.execute(query)
            orm_result = result.scalars().all()

        return [models.ClothesVector.from_orm(obj) for obj in orm_result]
