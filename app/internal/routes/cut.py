from fastapi import status, Depends, APIRouter

from app.pkg.models import CutTaskCmd, RabbitMQInfo
from app.internal.services.cut import CutService


cut_router = APIRouter(
    prefix="/cut",
    tags=["Cut"],
    responses={},
)

@cut_router.post(
    "/",
    response_model=CutTaskCmd,
    status_code=status.HTTP_201_CREATED,
    description="Create cut task.",
)
async def create_cut_task(
    cmd: CutTaskCmd,
    cut_service: CutService = Depends(CutService),
):
    return await cut_service.create_cut_task(cmd=cmd)

@cut_router.get(
    "/resp/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check response queue health info.",
)
async def resp_health(
    cut_service: CutService = Depends(CutService),
):
    return await cut_service.get_resp_repository_info()

@cut_router.get(
    "/task/health",
    response_model=RabbitMQInfo,
    status_code=status.HTTP_200_OK,
    description="Check task queue health info.",
)
async def task_health(
    cut_service: CutService = Depends(CutService),
):
    return await cut_service.get_task_repository_info()
