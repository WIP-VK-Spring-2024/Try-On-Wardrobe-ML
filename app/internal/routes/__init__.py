"""Global point for collected routers. __routes__ is a :class:`.Routes`
instance that contains all routers in your application.

Examples:
    After declaring all routers, you need to register them in your application::

        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> __routes__.register_routes(app=app)
"""

from app.pkg.models.core.routes import Routes
from app.internal.routes.try_on import try_on_router
from app.internal.routes.cut import cut_router
from app.internal.routes.outfit_gen import outfit_gen_router
from app.internal.routes.recsys import recsys_router
from app.internal.routes.postgres import postgres_router
from app.internal.routes.amazon_s3 import aws_router


__all__ = [
    "__routes__",
    "try_on_router",
    "cut_router",
    "outfit_gen_router",
    "recsys_router",
    "postgres_router",
    "aws_router",
]

__routes__ = Routes(
    routers=(
        try_on_router,
        cut_router,
        outfit_gen_router,
        recsys_router,

        postgres_router,
        aws_router,
    ),
)
