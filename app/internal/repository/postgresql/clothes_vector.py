"""Repository for clothes vector."""

from typing import List

from app.internal.repository.repository import Repository
from app.pkg.resources.postgresql.connection import get_connection
from app.internal.repository.postgresql.handlers.collect_response import (
    collect_response,
)
from app.pkg import models

__all__ = ["ClothesVectorRepository"]


class ClothesVectorRepository(Repository):
    """Clothes vector repository implementation."""

    @collect_response
    async def create(self, cmd: models.ClothesVectorCreateCmd) -> models.ClothesVector:
        q = """
            insert into clothes_vector(
                clothes_id, tensor
            ) values (
                %(clothes_id)s, %(tensor)s
            )
            returning id, clothes_id, tensor;
        """
        async with get_connection() as cur:
            await cur.execute(q, cmd.to_dict())
            return await cur.fetchone()

    @collect_response
    async def read_all(self) -> List[models.ClothesVector]:
        q = """
            select
                id, clothes_id, tensor
            from clothes_vector;
        """

        async with get_connection() as cur:
            await cur.execute(q)
            return await cur.fetchall()
