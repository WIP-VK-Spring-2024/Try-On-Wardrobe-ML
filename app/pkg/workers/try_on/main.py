"""Main module for start model task worker."""

import asyncio

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.pkg.workers.try_on.worker import TryOnWorker
from app.internal.services import AmazonS3Service
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator

__all__ = ["start_worker"]


def start_worker():
    task_repository = ModelTaskRepository()
    resp_repository = ModelRespRepository()
    file_service = AmazonS3Service()

    clothes_model = ClothProcessor()
    human_model = HumanProcessor()
    try_on_model = LadyVtonAggregator()

    model_worker = TryOnWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        file_service=file_service,
        clothes_model=clothes_model,
        human_model=human_model,
        try_on_model=try_on_model,
    )
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()