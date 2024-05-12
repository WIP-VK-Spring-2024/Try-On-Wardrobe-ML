from fastapi import status, Depends, APIRouter

from app.pkg.models import OutfitGenTaskCmd, RabbitMQInfo
from app.internal.services.outfit_gen import OutfitGenService


outfit_gen_router = APIRouter(
    prefix="/outfit_gen",
    tags=["Outfit gen"],
    responses={},
)

@outfit_gen_router.post(
    "/",
    response_model=OutfitGenTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create outfit gen task.",
)
async def create_outfit_gen_task(
    cmd: OutfitGenTaskCmd,
    outfit_gen_service: OutfitGenService = Depends(OutfitGenService),
):
    return await outfit_gen_service.create_outfit_gen_task(cmd=cmd)

@outfit_gen_router.get(
    "/resp/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check response queue health info.",
)
async def health(
    outfit_gen_service: OutfitGenService = Depends(OutfitGenService),
):
    return await outfit_gen_service.get_resp_repository_info()
