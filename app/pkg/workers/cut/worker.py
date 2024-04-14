"""Cut worker for read task queue."""

from typing import BinaryIO, Tuple

from app.internal.repository.rabbitmq.cut_task import CutTaskRepository
from app.internal.repository.rabbitmq.cut_response import CutRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import CutResponseCmd, ClothesTaskClassification, ClothesRespClassification
from app.pkg.logger import get_logger
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.autotags.autotag import AutoTagger
from app.pkg.settings import settings

logger = get_logger(__name__)

class CutWorker:
    """Model worker for read task queue."""

    task_repository: CutTaskRepository
    resp_repository: CutRespRepository
    file_service: AmazonS3Service
    clothes_model: ClothProcessor
    autotag: AutoTagger


    def __init__(
        self,
        task_repository: CutTaskRepository,
        resp_repository: CutRespRepository,
        file_service: AmazonS3Service,
        clothes_model: ClothProcessor,
        autotag_model: AutoTagger,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.file_service = file_service
    
        self.clothes_model = clothes_model
        self.autotag_model = autotag_model

    async def listen_queue(self):
        logger.info("Starting listen queue...")

        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            clothes_image = self.file_service.read(
                file_name=message.clothes_id,
                folder=message.clothes_dir,
            )

            logger.info(
                "Starting cut pipeline clothes id: [%s]",
                message.clothes_id,
            )
            # Model pipeline     
            cutted_clothes, classification = self.pipeline(
                clothes_image=clothes_image,
                tags=message.classification,
            )

            # Save result
            # TODO: check uuid to str
            res_file_name = message.clothes_id
            res_file_dir = settings.CUT_DIR

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
            )
            await self.resp_repository.create(cmd=cmd)

    def pipeline(self, clothes_image: BinaryIO, tags: ClothesTaskClassification) -> Tuple[BinaryIO, ClothesRespClassification]:
        # Remove model background
        cutted_clothes = self.clothes_model.consistent_forward(clothes_image)
        logger.debug("End removed background, result: [%s].", cutted_clothes["cloth"])
        
        input_data = {
            "image": clothes_image,
            "tags": tags.dict(),
        }
        classification = self.autotag_model.forward(input_data)
        logger.debug("End autotag model, result: [%s].", classification)

        return cutted_clothes["cloth"], classification