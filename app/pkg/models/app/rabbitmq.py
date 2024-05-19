"""Models of rabbitmq object."""

from typing import Optional

from pydantic.fields import Field

from app.pkg.models.base import BaseModel

__all__ = ["RabbitMQInfo"]

class BaseRabbitMQ(BaseModel):
    """Base rabbitmq model."""

class RabbitMQFields:
    """Model fields of rabbitmq model."""

    is_listening: bool = Field(
        default=False,
        description="Is queue has consumers or not.",
    )
    consumer_count: int = Field(
        default=0,
        description="Total consumers for queue.",
    )
    message_count: int = Field(
        default=0,
        description="Total messages in queue.",
    )
    response_time: float = Field(
        example=5.412,
        description="""
            Time elapsed for response if message sending
            right now in queue in seconds.
        """,
    )
    avg_response_time: float = Field(
        example=5.412,
        description="Average response time for one message in seconds.",
    )

class RabbitMQInfo(BaseRabbitMQ):
    consumer_count: int = RabbitMQFields.consumer_count
    message_count: int = RabbitMQFields.message_count
    avg_response_time: Optional[float] = RabbitMQFields.response_time
    is_listening: bool = RabbitMQFields.is_listening
    response_time: Optional[float] = RabbitMQFields.response_time
