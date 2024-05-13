from fastapi import status, Depends, APIRouter

from app.pkg.models import TryOnTaskCmd, RabbitMQInfo
from app.internal.services.try_on import TryOnService


try_on_router = APIRouter(
    prefix="/try_on",
    tags=["Try on"],
    responses={},
)

@try_on_router.post(
    "/",
    response_model=TryOnTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create try on task.",
)
async def create_try_on_task(
    cmd: TryOnTaskCmd,
    try_on_service: TryOnService = Depends(TryOnService),
):
    return await try_on_service.create_try_on_task(cmd=cmd)

@try_on_router.get(
    "/resp/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check response queue health info.",
)
async def resp_health(
    try_on_service: TryOnService = Depends(TryOnService),
):
    return await try_on_service.get_resp_repository_info()

@try_on_router.get(
    "/task/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check task queue health info.",
)
async def task_health(
    try_on_service: TryOnService = Depends(TryOnService),
):
    return await try_on_service.get_task_repository_info()
