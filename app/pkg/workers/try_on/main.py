"""Main module for start model task worker."""

import asyncio

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.pkg.workers.try_on.worker import TryOnWorker
from app.internal.services import AmazonS3Service
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor

__all__ = ["start_worker"]


def start_worker():
    task_repository = ModelTaskRepository()
    resp_repository = ModelRespRepository()
    file_service = AmazonS3Service()
    clothes_model = ClothProcessor()

    model_worker = TryOnWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        file_service=file_service,
        clothes_model=clothes_model,
    )
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()