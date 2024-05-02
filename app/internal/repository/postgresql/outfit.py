"""Repository for outfit."""

from typing import List

from app.internal.repository.repository import Repository
from app.pkg.resources.postgresql.connection import get_connection
from app.internal.repository.postgresql.handlers.collect_response import (
    collect_response,
)
from app.pkg import models

__all__ = ["OutfitRepository"]


class OutfitRepository(Repository):
    """Outfit repository implementation."""

    @collect_response
    async def read_all(self) -> List[models.Outfit]:
        q = """
            select * 
            from outfits;
        """

        async with get_connection() as cur:
            await cur.execute(q)
            return await cur.fetchall()

    @collect_response
    async def read_all_clothes_tensors(self) -> List[models.UserOutfitClothes]:
        q = """
            select
                outfits.id as outfit_id,
                outfits.user_id,
                array_agg(cv.clothes_id) as clothes,
                array_agg(cv.tensor) as clothes_tensor
            from outfits
            join clothes_vector cv on outfits.transforms ? cv.clothes_id::text
            where outfits.privacy = 'public'
            group by
                outfits.id,
                outfits.user_id;
        """

        async with get_connection() as cur:
            await cur.execute(q)
            return await cur.fetchall()
