"""Rec sys worker for read task queue."""

from fastapi import status

from app.internal.repository.rabbitmq.recsys_task import RecSysTaskRepository
from app.internal.repository.rabbitmq.recsys_response import RecSysRespRepository
from app.internal.repository.postgresql.outfit import OutfitRepository
from app.pkg.models import RecSysResponseCmd
from app.pkg.logger import get_logger
from app.pkg.ml.outfits_recsys.recsys import CrossUsersOutfitRecSys

logger = get_logger(__name__)

class RecSysWorker:
    """Model worker for read task queue."""

    task_repository: RecSysTaskRepository
    resp_repository: RecSysRespRepository
    recsys_model: CrossUsersOutfitRecSys
    outfit_repository: OutfitRepository


    def __init__(
        self,
        task_repository: RecSysTaskRepository,
        resp_repository: RecSysRespRepository,
        outfit_repository: OutfitRepository,
        recsys_model: CrossUsersOutfitRecSys,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.outfit_repository = outfit_repository
    
        self.recsys_model = recsys_model

    async def update_recsys_model_data(self):
        logger.info("RecSys first initializing system with new outfits...")
        clothes_tensors = await self.outfit_repository.read_all_clothes_tensors()

        # Получаем необходимые форматы данных для модели
        outfits = [clothes.dict(exclude={'clothes_tensor'}) for clothes in clothes_tensors]

        cloth_id_to_list = {}
        for outfit in clothes_tensors:
            for cloth_id, tensor in zip(outfit.clothes, outfit.clothes_tensor):
                cloth_id_to_list[cloth_id] = tensor

        logger.debug("Total outfits: [%s]", len(outfits))
        self.recsys_model.update_global_outfits_list_format(
            outfits=outfits,
            cloth_id_to_list=cloth_id_to_list,
        )
        logger.info("RecSys model successfully initialized.")

    async def listen_queue(self):
        await self.update_recsys_model_data()

        logger.info("Starting listen queue...")

        async for message in self.task_repository.read():
            logger.info(
                "New message user id: [%s], samples amount: [%s]",
                message.user_id,
                message.samples_amount,
            )

            logger.info("Starting recsys pipeline")
            # Model pipeline
            try:
                outfit_ids = self.recsys_model.recommend(
                    user_id=message.user_id,
                    samples=message.samples_amount,
                )
                cmd = RecSysResponseCmd(
                    user_id=message.user_id,
                    outfit_ids=outfit_ids,
                    status_code=status.HTTP_200_OK,
                    message=f"Successfully generated {len(outfit_ids)} clothes.",
                )
                logger.debug("End pipeline, generated clothes %s: [%s]", len(outfit_ids), outfit_ids)
            except Exception as exc:
                logger.exception("Pipeline error type: [%s], error: [%s]", type(exc), exc)
                cmd = RecSysResponseCmd(user_id=message.user_id, message=str(exc))

            logger.info("Result model: [%s]", cmd)
            await self.resp_repository.create(cmd=cmd)
