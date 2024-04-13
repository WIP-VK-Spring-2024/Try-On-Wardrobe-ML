"""Rec sys worker for read task queue."""

from io import BytesIO
from typing import BinaryIO, List, Dict
from uuid import UUID4

from app.internal.repository.rabbitmq.outfit_gen_task import OutfitGenTaskRepository
from app.internal.repository.rabbitmq.outfit_gen_response import OutfitGenRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import OutfitGenClothes, OutfitGenResponseCmd, ImageCategoryAutoset
from app.pkg.logger import get_logger
from app.pkg.ml.auto_clothing_set.autoset import LocalRecSys
from app.pkg.settings import settings

logger = get_logger(__name__)

class OutfitGenWorker:
    """Model worker for read task queue."""

    task_repository: OutfitGenTaskRepository
    resp_repository: OutfitGenRespRepository
    file_service: AmazonS3Service
    outfit_gen_model: LocalRecSys


    def __init__(
        self,
        task_repository: OutfitGenTaskRepository,
        resp_repository: OutfitGenRespRepository,
        file_service: AmazonS3Service,
        outfit_gen_model: LocalRecSys
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.file_service = file_service
    
        self.outfit_gen_model = outfit_gen_model

    async def listen_queue(self):
        logger.info("Starting listen queue...")

        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            data = self.prepare_data(
                message.clothes,
                folder=settings.CLOTHES_DIR,
            )
            print(data)

            # logger.info(
            #     "Starting try on pipeline, %s clothes: [%s]",
            #     len(message.clothes),
            #     message.clothes,
            # )
            # # Model pipeline           
            # try_on = self.pipeline(
            #     user_image=user_image,
            #     clothes=message.clothes,
            #     clothes_images=clothes_images,
            # )

            # # Save result
            # res_file_name = f"{message.clothes[0].clothes_id}"
            # res_file_dir = f"{settings.ML.TRY_ON_DIR}/{message.user_image_id}"

            # self.file_service.upload(
            #     file=try_on,
            #     file_name=res_file_name,
            #     folder=res_file_dir,
            # )
            
            # logger.info(
            #     "Try on result file name [%s], dir [%s]",
            #     res_file_name,
            #     res_file_dir,
            # )
            cmd = OutfitGenResponseCmd(
                **message.dict(),
            )
            await self.resp_repository.create(cmd=cmd)

    def read_clothes(
        self,
        clothes: List[OutfitGenClothes],
        folder: str
    ) -> Dict[ImageCategoryAutoset, List[Dict[Dict[str, UUID4], Dict[str, BytesIO]]]]:
        """Read clothes from file service in correct order"""
        result = {
            ImageCategoryAutoset.UPPER_BODY: [],
            ImageCategoryAutoset.LOWER_BODY: [],
            ImageCategoryAutoset.DRESSES: [],
            ImageCategoryAutoset.OUTWEAR: [],
        }

        for clothe in clothes:
            category = ImageCategoryAutoset(clothe.category)
            image = self.file_service.read(
                file_name=clothe.clothes_id,
                folder=folder,
            )
            clothe = {
                'clothe': image,
                'clothes_id': clothe.clothes_id,
            }

            result[ImageCategoryAutoset].append(clothe)
    
        return result
    

    # def pipeline(
    #     self,
    #     user_image: BinaryIO,
    #     clothes: List[TryOnClothes],
    #     clothes_images: List[BytesIO],
    # ) -> BytesIO:
    #     """Try on model pipeline
        
    #     Args:
    #         clothes_images: List[TryOnImageClothes], list of clothes images to try on
    #         user_image: BinaryIO, user image for processing
    #     """
    #     # Human processing
    #     processed_user = self.human_model.consistent_forward(user_image)
    #     logger.debug("End human processing, result: [%s]", processed_user["parse_orig"])
        
    #     # Try on
    #     try_on_clothes = [
    #         {
    #             "category": ImageCategory(clothe.category),
    #             "cloth": image,
    #         } 
    #         for clothe, image in zip(clothes, clothes_images)
    #     ]

    #     try_on = self.try_on_model.try_on_set(
    #         human=processed_user,
    #         clothes=try_on_clothes,
    #     )
    #     logger.info("End try on, result: [%s]", try_on)

    #     return try_on