"""Handle Postgresql Query Exceptions."""

from typing import Callable

import psycopg2

from app.pkg.models.base import BaseModel
from app.pkg.models.exceptions.association import __aiopg__, __constrains__
from app.pkg.models.exceptions.repository import DriverError
from app.pkg.logger import get_logger

__all__ = ["handle_exception"]

logger = get_logger(__name__)

def handle_exception(func: Callable[..., BaseModel]):
    """Decorator Catching Postgresql Query Exceptions.

    Args:
        func:
            callable function object.

    Raises:
        UniqueViolation: The query violates the domain uniqueness constraints
            of the database set.
        DriverError: Any error during execution query on a database.
    """

    async def wrapper(*args: object, **kwargs: object) -> BaseModel:
        try:
            return await func(*args, **kwargs)
        except psycopg2.Error as error:
            if exc := __constrains__.get(error.diag.constraint_name):
                raise exc from error

            if exc := __aiopg__.get(error.pgcode):
                raise exc from error

            logger.error("Psycopg2 error: [%s]", error)
            raise DriverError(details=error.diag.message_detail) from error

    return wrapper
