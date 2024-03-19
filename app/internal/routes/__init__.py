"""Global point for collected routers. __routes__ is a :class:`.Routes`
instance that contains all routers in your application.

Examples:
    After declaring all routers, you need to register them in your application::

        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> __routes__.register_routes(app=app)
"""

from app.pkg.models.core.routes import Routes
from app.internal.routes.model import model_router


__all__ = [
    "__routes__",
    "model_router",
]

__routes__ = Routes(
    routers=(
        model_router,
    ),
)
