"""Main module for start recsys model task worker."""

import asyncio

from app.internal.repository.rabbitmq.outfit_gen_task import OutfitGenTaskRepository
from app.internal.repository.rabbitmq.outfit_gen_response import OutfitGenRespRepository
from app.pkg.workers.outfit_gen.worker import OutfitGenWorker
from app.internal.services import AmazonS3Service
from app.pkg.ml.auto_clothing_set.autoset import LocalRecSys
from app.pkg.logger import get_logger

logger = get_logger(__name__)

__all__ = ["start_worker"]


def start_worker():
    logger.info("Starting initialization...")
    task_repository = OutfitGenTaskRepository()
    resp_repository = OutfitGenRespRepository()
    file_service = AmazonS3Service()

    outfit_gen_model = LocalRecSys()

    model_worker = OutfitGenWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        file_service=file_service,
        outfit_gen_model=outfit_gen_model,
    )
    logger.info("Successfuly initializated.")
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()