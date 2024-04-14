"""Try on worker for read task queue."""

from io import BytesIO
from typing import BinaryIO, List

from app.internal.repository.rabbitmq.try_on_task import TryOnTaskRepository
from app.internal.repository.rabbitmq.try_on_response import TryOnRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import TryOnResponseCmd, TryOnClothes, ImageCategory
from app.pkg.logger import get_logger
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator
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

            user_image = self.file_service.read(
                file_name=message.user_image_id,
                folder=message.user_image_dir,
            )

            clothes_images = self.read_clothes(message.clothes, folder=settings.CLOTHES_DIR)

            logger.info(
                "Starting try on pipeline, %s clothes: [%s]",
                len(message.clothes),
                message.clothes,
            )
            # Model pipeline           
            try_on = self.pipeline(
                user_image=user_image,
                clothes=message.clothes,
                clothes_images=clothes_images,
            )

            # Save result
            res_file_name = f"{message.clothes[0].clothes_id}"
            res_file_dir = f"{settings.TRY_ON_DIR}/{message.user_image_id}"

            self.file_service.upload(
                file=try_on,
                file_name=res_file_name,
                folder=res_file_dir,
            )
            
            logger.info(
                "Try on result file name [%s], dir [%s]",
                res_file_name,
                res_file_dir,
            )
            cmd = TryOnResponseCmd(
                **message.dict(),
                try_on_id=res_file_name,
                try_on_dir=res_file_dir,
            )
            await self.resp_repository.create(cmd=cmd)

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