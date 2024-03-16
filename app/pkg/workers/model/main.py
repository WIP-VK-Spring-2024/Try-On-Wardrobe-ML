"""Main module for start model task worker."""

import asyncio

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.pkg.workers.model.worker import ModelWorker
from app.internal.services import FileService

__all__ = ["start_worker"]


def start_worker():
    task_repository = ModelTaskRepository()
    resp_repository = ModelRespRepository()
    file_service = FileService()

    model_worker = ModelWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        file_service=file_service,
    )
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()