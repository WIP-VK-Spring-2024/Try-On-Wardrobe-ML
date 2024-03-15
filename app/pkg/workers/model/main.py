"""Main module for start model task worker."""

import asyncio

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.pkg.workers.model.worker import ModelWorker

__all__ = ["start_worker"]


def start_worker():
    task_repository = ModelTaskRepository()
    resp_repository = ModelRespRepository()

    model_worker = ModelWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
    )
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()