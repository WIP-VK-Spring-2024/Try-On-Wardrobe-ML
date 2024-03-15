"""Main module for start model task worker."""

import asyncio

from app.internal.repository.rabbitmq.model import ModelRepository
from app.pkg.workers.model.worker import ModelWorker

__all__ = ["start_worker"]


def start_worker():
    rabbit_repository = ModelRepository()
    model_worker = ModelWorker(rabbit_repository)
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()