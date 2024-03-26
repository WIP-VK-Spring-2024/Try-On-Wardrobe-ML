"""Handle Rabbitmq Query Exceptions."""

from typing import Callable

import aio_pika

from app.pkg.models.base import BaseModel
from app.pkg.models.exceptions.repository import (
    DriverError,
    EmptyResult,
)

__all__ = ["handle_exception"]


def handle_exception(func: Callable[..., BaseModel]):
    async def wrapper(*args: object, **kwargs: object) -> BaseModel:
        try:
            return await func(*args, **kwargs)
        except aio_pika.exceptions.AMQPError as error:
            raise DriverError(details=str(error)) from error

    return wrapper
