"""Model worker for read task queue."""

from app.internal.repository.rabbitmq.model import ModelRepository

class ModelWorker:
    """Model worker for read task queue."""

    # rabbit_repository: RabbitRepository

    def __init__(
        self,
        rabbit_repository: ModelRepository,
    ):
        self.rabbit_repository = rabbit_repository

    async def listen_queue(self):
        async for message in self.rabbit_repository.read():
            print(message)