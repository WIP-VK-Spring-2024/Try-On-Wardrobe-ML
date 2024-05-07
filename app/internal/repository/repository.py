"""Abstract repository interface."""

from abc import ABC
from typing import List, TypeVar

from app.pkg.models.base import BaseModel

__all__ = ["Repository", "BaseRepository"]

BaseRepository = TypeVar("BaseRepository", bound="Repository")


class Repository(ABC):
    async def create(self, cmd: BaseModel) -> BaseModel:
        """Create model.

        Args:
            cmd (Model): Specific command for create model. Must be inherited from
                ``Model``.

        Returns:
            Type of the parent model.
        """
        raise NotImplementedError

    async def read(self, query: BaseModel) -> BaseModel:
        """Read model.

        Args:
            query (Model): Specific query for read model. Must be inherited from
                ``Model``.

        Returns:
            Type of the parent model.
        """

        raise NotImplementedError

    async def read_all(self) -> List[BaseModel]:
        """Read all rows."""

        raise NotImplementedError

    async def update(self, cmd: BaseModel) -> BaseModel:
        """Update model.

        Notes: In this method cmd must contain id of the model for update and ALL
        fields for update.

        Returns:
            Type of the parent model.
        """

        raise NotImplementedError

    async def delete(self, cmd: BaseModel) -> BaseModel:
        """Delete model.

        Notes: In this method you should mark row as deleted. You must not delete row
            from database.

        Returns:
            Type of the parent model.
        """

        raise NotImplementedError
