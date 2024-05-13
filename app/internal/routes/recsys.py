from fastapi import status, Depends, APIRouter

from app.pkg.models import RecSysTaskCmd, RabbitMQInfo
from app.internal.services.recsys import RecSysService


recsys_router = APIRouter(
    prefix="/recsys",
    tags=["Rec sys"],
    responses={},
)

@recsys_router.post(
    "/",
    response_model=RecSysTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create rec sys task.",
)
async def create_recsys_task(
    cmd: RecSysTaskCmd,
    recsys_service: RecSysService = Depends(RecSysService),
):
    return await recsys_service.create_recsys_task(cmd=cmd)

@recsys_router.get(
    "/resp/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check response queue health info.",
)
async def resp_health(
    recsys_service: RecSysService = Depends(RecSysService),
):
    return await recsys_service.get_resp_repository_info()

@recsys_router.get(
    "/task/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check task queue health info.",
)
async def task_health(
    recsys_service: RecSysService = Depends(RecSysService),
):
    return await recsys_service.get_task_repository_info()
