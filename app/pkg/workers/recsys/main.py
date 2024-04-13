"""Main module for start recsys model task worker."""

import asyncio

from app.internal.repository.rabbitmq.recsys_task import RecSysTaskRepository
from app.internal.repository.rabbitmq.recsys_response import RecSysRespRepository
from app.pkg.workers.recsys.worker import RecSysWorker
from app.internal.services import AmazonS3Service
# from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
# from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
# from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator
from app.pkg.logger import get_logger

logger = get_logger(__name__)

__all__ = ["start_worker"]


def start_worker():
    logger.info("Starting initialization...")
    task_repository = RecSysTaskRepository()
    resp_repository = RecSysRespRepository()
    file_service = AmazonS3Service()

    # clothes_model = ClothProcessor()
    # human_model = HumanProcessor()
    # try_on_model = LadyVtonAggregator()

    model_worker = RecSysWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        file_service=file_service,
        # clothes_model=clothes_model,
        # human_model=human_model,
        # try_on_model=try_on_model,
    )
    logger.info("Successfuly initializated.")
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()