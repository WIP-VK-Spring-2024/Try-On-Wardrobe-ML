"""Model worker for read task queue."""

from app.internal.repository.rabbitmq.model_task import ModelTaskRepository
from app.internal.repository.rabbitmq.model_response import ModelRespRepository
from app.internal.services import FileService
from app.pkg.models import CreateRespFileCmd
from app.pkg.logger import get_logger

logger = get_logger(__name__)

class ModelWorker:
    """Model worker for read task queue."""

    # rabbit_repository: RabbitRepository

    def __init__(
        self,
        task_repository: ModelTaskRepository,
        resp_repository: ModelRespRepository,
        file_service: FileService,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.file_service = file_service

    async def listen_queue(self):
        async for message in self.task_repository.read():
            logger.info("New message [%s]", message)

            # res_file_name, res_file_path = self.file_service.get_file_path(
            #     extension='png',
            # )
            res_file_name, res_file_path = self.file_service.get_mock_file_path(file_id=message.clothes_id)
            # try_on_file_path = await try_on_model.pipeline(try_on_file_path)
            
            logger.info(
                "New message generated file name [%s], file path [%s]",
                res_file_name,
                res_file_path,
            )
            cmd = CreateRespFileCmd(
                clothes_id=message.clothes_id,
                user_id=message.user_id,
                res_file_name=res_file_name,
                res_file_path=str(res_file_path),
            )
            await self.resp_repository.create(cmd=cmd)
