"""Main module for start cut model task worker."""

import asyncio

from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.cut_response import CutRespRepository
from app.internal.repository.postgresql.clothes_vector import ClothesVectorRepository
from app.pkg.workers.cut.worker import CutWorker
from app.internal.services import AmazonS3Service
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.autotags.autotag import AutoTagger
from app.pkg.ml.auto_clothing_set.autoset import LocalRecSys
from app.pkg.logger import get_logger
from app.pkg.settings import settings

logger = get_logger(__name__)

__all__ = ["start_worker"]


def start_worker():
    logger.info("Starting initialization...")
    task_repository = CutTaskRepository()
    resp_repository = CutRespRepository()
    clothes_vector_repository = ClothesVectorRepository()

    file_service = AmazonS3Service()

    clothes_model = ClothProcessor(light_weight=settings.ML.IS_CUT_LIGHT_WEIGHT)
    autotag_model = AutoTagger()
    vectorizer_model = LocalRecSys()

    model_worker = CutWorker(
        task_repository=task_repository,
        resp_repository=resp_repository,
        clothes_vector_repository=clothes_vector_repository,
        file_service=file_service,
        clothes_model=clothes_model,
        autotag_model=autotag_model,
        vectorizer_model=vectorizer_model,
    )
    logger.info("Successfuly initializated.")
    asyncio.run(model_worker.listen_queue())


if __name__ == "__main__":
    start_worker()