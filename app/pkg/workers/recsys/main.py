"""Main module for start recsys model task worker."""

import asyncio

from app.internal.repository.rabbitmq.recsys_task import RecSysTaskRepository
from app.internal.repository.rabbitmq.recsys_response import RecSysRespRepository
from app.pkg.workers.recsys.worker import RecSysWorker
from app.pkg.ml.outfits_recsys.recsys import CrossUsersOutfitRecSys
from app.pkg.logger import get_logger

logger = get_logger(__name__)

__all__ = ["start_worker"]


def start_worker():
    logger.info("Starting initialization...")
    task_repository = RecSysTaskRepository()
    resp_repository = RecSysRespRepository()

    recsys_model = CrossUsersOutfitRecSys()

    model_worker = RecSysWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        recsys_model=recsys_model,
    )
    logger.info("Successfuly initializated.")
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()