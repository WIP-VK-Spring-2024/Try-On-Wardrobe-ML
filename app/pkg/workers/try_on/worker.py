"""Try on worker for read task queue."""

import uuid
from io import BytesIO
from typing import BinaryIO, List

from fastapi import status

from app.internal.repository.rabbitmq.try_on_task import TryOnTaskRepository
from app.internal.repository.rabbitmq.try_on_response import TryOnRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import TryOnTaskCmd, TryOnResponseCmd, TryOnClothes, ImageCategory
from app.pkg.logger import get_logger
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator
from app.pkg.models.exceptions.amazon_s3 import AmazonS3Error
from app.pkg.settings import settings

logger = get_logger(__name__)

class TryOnWorker:
    """Model worker for read task queue."""

    task_repository: TryOnTaskRepository
    resp_repository: TryOnRespRepository
    file_service: AmazonS3Service
    clothes_model: ClothProcessor
    human_model = HumanProcessor
    try_on_model = LadyVtonAggregator


    def __init__(
        self,
        task_repository: TryOnTaskRepository,
        resp_repository: TryOnRespRepository,
        file_service: AmazonS3Service,
        clothes_model: ClothProcessor,
        human_model: HumanProcessor,
        try_on_model: LadyVtonAggregator,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.file_service = file_service
    
        self.clothes_model = clothes_model
        self.human_model = human_model
        self.try_on_model = try_on_model

    async def listen_queue(self):
        logger.info("Starting listen queue...")

        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            try:
                user_image = self.file_service.read(
                    file_name=message.user_image_id,
                    folder=message.user_image_dir,
                )
                clothes_images = self.read_clothes(message.clothes, folder=settings.CUT_DIR)
            except AmazonS3Error as exc:
                logger.error("Amazon s3 read error, clothes: [%s], error: [%s]", message.clothes, exc)
                cmd = TryOnResponseCmd(
                    **message.dict(),
                    status_code=exc.status_code,
                    message=exc.message,
                )
                await self.resp_repository.create(cmd=cmd)
                continue

            logger.info(
                "Starting try on pipeline, %s clothes: [%s]",
                len(message.clothes),
                message.clothes,
            )
            # Model pipeline
            try:
                try_on_image = self.pipeline(
                    user_image=user_image,
                    clothes=message.clothes,
                    clothes_images=clothes_images,
                )
            except Exception as exc:
                logger.error("Pipeline error type: [%s], error: [%s]", type(exc), exc)
                cmd = TryOnResponseCmd(
                    **message.dict(),
                    message=str(exc),
                )
                await self.resp_repository.create(cmd=cmd)
                continue

            # Save result
            await self.save_result(message=message, try_on_image=try_on_image)

    def read_clothes(self, clothes: List[TryOnClothes], folder: str) -> List[BytesIO]:
        """Read clothes from file service"""
        images = []
        for clothe in clothes:
            images.append(
                self.file_service.read(
                    file_name=clothe.clothes_id,
                    folder=folder,
                ),
            )
        return images

    def pipeline(
        self,
        user_image: BinaryIO,
        clothes: List[TryOnClothes],
        clothes_images: List[BytesIO],
    ) -> BytesIO:
        """Try on model pipeline
        
        Args:
            clothes_images: List[TryOnImageClothes], list of clothes images to try on
            user_image: BinaryIO, user image for processing
        """
        # Human processing
        processed_user = self.human_model.consistent_forward(user_image)
        logger.debug("End human processing, result: [%s]", processed_user["parse_orig"])
        
        # Try on
        try_on_clothes = [
            {
                "category": ImageCategory(clothe.category),
                "cloth": image,
            } 
            for clothe, image in zip(clothes, clothes_images)
        ]

        try_on = self.try_on_model.try_on_set(
            human=processed_user,
            clothes=try_on_clothes,
        )
        logger.info("End try on, result: [%s]", try_on)

        return try_on
    
    async def save_result(self, message: TryOnTaskCmd, try_on_image: BytesIO) -> None:
        res_file_name = str(message.outfit_id) if message.outfit_id else str(uuid.uuid4())
        res_file_dir = f"{settings.TRY_ON_DIR}/{message.user_image_id}"
        logger.info(
            "Try on result file name [%s], dir [%s]",
            res_file_name,
            res_file_dir,
        )
        try:
            self.file_service.upload(
                file=try_on_image,
                file_name=res_file_name,
                folder=res_file_dir,
            )

            cmd = TryOnResponseCmd(
                **message.dict(),
                try_on_id=res_file_name,
                try_on_dir=res_file_dir,
                status_code=status.HTTP_201_CREATED,
                message="Successfully created try on image.",
            )
        except AmazonS3Error as exc:
            logger.error("Amazon s3 upload error: [%s]", exc)
            cmd = TryOnResponseCmd(
                **message.dict(),
                status_code=exc.status_code,
                message=exc.message,
            )

        await self.resp_repository.create(cmd=cmd)