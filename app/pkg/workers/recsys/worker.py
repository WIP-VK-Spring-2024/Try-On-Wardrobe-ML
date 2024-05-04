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
        recsys_model: CrossUsersOutfitRecSys,
        outfit_repository: OutfitRepository,
    ):
        self.task_repository = task_repository
        self.resp_repository = resp_repository
        self.outfit_repository = outfit_repository
    
        self.recsys_model = recsys_model

    async def update_recsys_model_data(self):
        logger.info("RecSys first initializing system with new outfits...")
        clothes_tensors = await self.outfit_repository.read_all_clothes_tensors()

        # Получаем необходимые форматы данных для модели
        outfits = clothes_tensors.dict(exclude={'clothes_tensor'})

        cloth_id_to_bytes = {}
        for clothes in clothes_tensors:
            for cloth_id, tensor in zip(clothes.clothes, clothes.clothes_tensor):
                cloth_id_to_bytes[cloth_id] = tensor

        self.recsys_model.update_global_outfits_from_bytes(
            outfits=outfits,
            cloth_id_to_bytes=cloth_id_to_bytes,
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
                clothes_ids = self.recsys_model.recommend(
                    user_id=message.user_id,
                    samples=message.samples_amount,
                )
                cmd = RecSysResponseCmd(
                    clothes_ids=clothes_ids,
                    status_code=status.HTTP_200_OK,
                    message=f"Successfully generated {len(clothes_ids)} clothes.",
                )
                logger.debug("End pipeline, generated clothes %s: [%s]", len(clothes_ids), clothes_ids)
            except Exception as exc:
                logger.error("Pipeline error type: [%s], error: [%s]", type(exc), exc)
                cmd = RecSysResponseCmd(message=str(exc))

            logger.info("Result model: [%s]", cmd)
            await self.resp_repository.create(cmd=cmd)
