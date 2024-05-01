"""Handle SQLalchemy Query Exceptions."""

from typing import Callable

from sqlalchemy.exc import SQLAlchemyError

from app.pkg.models.base import BaseModel
from app.pkg.models.exceptions.repository import DriverError
from app.pkg.logger import get_logger

__all__ = ["handle_exception"]

logger = get_logger(__name__)

def handle_exception(func: Callable[..., BaseModel]):
    async def wrapper(*args: object, **kwargs: object) -> BaseModel:
        try:
            return await func(*args, **kwargs)
        except SQLAlchemyError as error:
            logger.error("SQLAlchemy error: [%s]", error)
            raise DriverError(details=str(error)) from error

    return wrapper
