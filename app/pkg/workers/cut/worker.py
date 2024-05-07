"""Cut worker for read task queue."""

import copy
from typing import BinaryIO, Tuple

from fastapi import status
from pydantic import UUID4

from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.cut_response import CutRespRepository
from app.internal.repository.postgresql.clothes_vector import ClothesVectorRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import (
    CutTaskCmd,
    CutResponseCmd,
    ClothesTaskClassification,
    ClothesRespClassification,
    ClothesVectorCreateCmd,
)
from app.pkg.logger import get_logger
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.autotags.autotag import AutoTagger
from app.pkg.models.exceptions.amazon_s3 import AmazonS3Error
from app.pkg.ml.auto_clothing_set.autoset import LocalRecSys
from app.pkg.settings import settings

logger = get_logger(__name__)

class CutWorker:
    """Model worker for read task queue."""

    task_repository: CutTaskRepository
    resp_repository: CutRespRepository
    clothes_vector_repository: ClothesVectorRepository
    file_service: AmazonS3Service
    clothes_model: ClothProcessor
    autotag: AutoTagger
    vectorizer_model: LocalRecSys


    def __init__(
        self,
        task_repository: CutTaskRepository,
        resp_repository: CutRespRepository,
        clothes_vector_repository: ClothesVectorRepository,
        file_service: AmazonS3Service,
        clothes_model: ClothProcessor,
        autotag_model: AutoTagger,
        vectorizer_model: LocalRecSys,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.clothes_vector_repository = clothes_vector_repository
        self.file_service = file_service
    
        self.clothes_model = clothes_model
        self.autotag_model = autotag_model
        self.vectorizer_model = vectorizer_model

    async def listen_queue(self):
        logger.info("Starting listen queue...")

        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            try:
                clothes_image = self.file_service.read(
                    file_name=message.clothes_id,
                    folder=message.clothes_dir,
                )
            except AmazonS3Error as exc:
                logger.error(
                    "Amazon s3 read error, clothe dir: [%s], id: [%s], error: [%s]",
                    message.clothes_dir,
                    message.clothes_id,
                    exc,
                )
                cmd = CutResponseCmd(
                    user_id=message.user_id,
                    clothes_id=message.clothes_id,
                    status_code=exc.status_code,
                    message=exc.message,
                )
                await self.resp_repository.create(cmd=cmd)
                continue

            logger.info(
                "Starting cut pipeline clothes id: [%s]",
                message.clothes_id,
            )
            # Model pipeline
            try:
                cutted_clothes, classification = await self.pipeline(
                    clothes_id=message.clothes_id,
                    clothes_image=clothes_image,
                    tags=message.classification,
                )
            except Exception as exc:
                logger.exception("Pipeline error type: [%s], error: [%s]", type(exc), exc)
                cmd = CutResponseCmd(
                    user_id=message.user_id,
                    clothes_id=message.clothes_id,
                    message=str(exc),
                )
                await self.resp_repository.create(cmd=cmd)
                continue

            # Save result
            await self.save_results(message, cutted_clothes, classification)

    async def pipeline(self, clothes_id: UUID4, clothes_image: BinaryIO, tags: ClothesTaskClassification) -> Tuple[BinaryIO, ClothesRespClassification]:
        # Remove model background
        cutted_clothes = self.clothes_model.consistent_forward(clothes_image)
        logger.debug("End removed background, result: [%s].", cutted_clothes["cloth"])
        
        input_data = {
            "image": clothes_image,
            "tags": tags.dict(),
        }
        classification = self.autotag_model.forward(input_data)
        logger.debug("End autotag model, result: [%s].", classification)
        
        # TODO: Add vectorizer to background
        cloth_image = copy.copy(cutted_clothes["cloth"])
        clothes_embs = self.vectorizer_model.get_embs_for_clothes([cutted_clothes])
        clothes_tensor = clothes_embs[0]['tensor']
        
        cmd = ClothesVectorCreateCmd(clothes_id=clothes_id, tensor=clothes_tensor)
        await self.clothes_vector_repository.create(cmd)
        logger.debug("End vectorizer, result: [%s].", clothes_tensor)

        return cloth_image, classification

    async def save_results(
        self,
        message: CutTaskCmd,
        cutted_clothes: BinaryIO,
        classification: ClothesRespClassification,
    ) -> None:
        res_file_name = message.clothes_id
        res_file_dir = settings.CUT_DIR

        try:
            self.file_service.upload(
                file=cutted_clothes,
                file_name=res_file_name,
                folder=res_file_dir,
            )
            
            logger.info(
                "Cut result dir [%s], file name [%s], classification [%s]",
                res_file_dir,
                res_file_name,
                classification,
            )
            cmd = CutResponseCmd(
                user_id=message.user_id,
                clothes_id=message.clothes_id,
                clothes_dir=res_file_dir,
                classification=classification,
                status_code=status.HTTP_201_CREATED,
                message="Successfully created cut image with classification.",
            )
        except AmazonS3Error as exc:
            logger.error("Amazon s3 upload error: [%s]", exc)
            cmd = CutResponseCmd(
                user_id=message.user_id,
                clothes_id=message.clothes_id,
                status_code=exc.status_code,
                message=exc.message,
            )

        await self.resp_repository.create(cmd=cmd)