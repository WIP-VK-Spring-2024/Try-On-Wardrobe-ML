"""Model worker for read task queue."""

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository

class ModelWorker:
    """Model worker for read task queue."""

    # rabbit_repository: RabbitRepository

    def __init__(
        self,
        task_repository: ModelTaskRepository,
        resp_repository: ModelRespRepository,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository

    async def listen_queue(self):
        async for message in self.task_repository.read():
            print(message)
            await self.resp_repository.create(cmd=message)