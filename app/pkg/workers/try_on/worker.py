"""Try on worker for read task queue."""

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import TryOnResponseCmd
from app.pkg.logger import get_logger
from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator

logger = get_logger(__name__)

class TryOnWorker:
    """Model worker for read task queue."""

    task_repository: ModelTaskRepository
    resp_repository: ModelRespRepository
    file_service: AmazonS3Service
    clothes_model: ClothProcessor
    human_model = HumanProcessor
    try_on_model = LadyVtonAggregator


    def __init__(
        self,
        task_repository: ModelTaskRepository,
        resp_repository: ModelRespRepository,
        file_service: AmazonS3Service,
        clothes_model: ClothProcessor,
        human_model = HumanProcessor,
        try_on_model = LadyVtonAggregator,
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

            clothes_image = self.file_service.read(
                file_name=message.clothes_id,
                folder=message.clothes_dir,
            )

            logger.info(
                "Started removing background clothes id: [%s]",
                message.clothes_id,
            )

            # Model pipeline           
            # Remove model background
            cutted_clothes = self.clothes_model.consistent_forward(clothes_image)
            logger.debug("End remove background, result: [%s].", cutted_clothes["cloth"])
            
            # Human processing
            processed_user = self.human_model.consistent_forward(user_image)
            logger.debug("End human processing, result: [%s]", processed_user["parse_orig"])
            
            # Try on
            processed_user.update(cutted_clothes)
            processed_user.update({"category": "upper_body"}) # TODO: add to enum
            try_on = self.try_on_model(processed_user)
            logger.debug("End try on, result: [%s]", try_on)
            # End model pipeline

            res_file_name = message.clothes_id
            res_file_dir = f"try_on/{message.user_image_id}" # TODO: add path to settings

            res_file_path = f"{res_file_dir}/{res_file_name}"

            self.file_service.upload(
                file=try_on,
                file_name=res_file_name,
                folder=res_file_dir,
            )
            
            logger.info(
                "Try on result file name [%s], path [%s]",
                res_file_name,
                res_file_path,
            )
            cmd = TryOnResponseCmd(
                **message.dict(),
                try_on_result_path=res_file_path,
            )
            await self.resp_repository.create(cmd=cmd)
