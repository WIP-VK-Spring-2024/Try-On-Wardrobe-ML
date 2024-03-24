"""Model worker for read task queue."""

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.internal.services import AmazonS3Service
from app.pkg.models import TryOnResponseCmd
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class ModelWorker:
    """Model worker for read task queue."""

    task_repository: ModelTaskRepository
    resp_repository: ModelRespRepository
    file_service: AmazonS3Service

    def __init__(
        self,
        task_repository: ModelTaskRepository,
        resp_repository: ModelRespRepository,
        file_service: AmazonS3Service,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.file_service = file_service

    async def listen_queue(self):
        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            user_image, user_image_path = self.file_service.read_and_save(
                file_name=message.user_image_id,
                folder=message.user_image_dir,
            )

            clothes_image, clothes_image_path = self.file_service.read_and_save(
                file_name=message.clothes_id,
                folder=message.clothes_dir,
            )
            logger.info(
                "New message local saved user file: [%s], clothes file: [%s]",
                user_image_path,
                clothes_image_path,
            )

            # try_on_file_path = await try_on_model.pipeline(try_on_file_path)

            res_file_name = message.clothes_id
            res_file_dir = f"try_on/{message.user_image_id}" # TODO: add path to settings

            res_file_path = f"{res_file_dir}/{res_file_name}"

            self.file_service.upload(
                file=clothes_image,
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
