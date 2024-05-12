"""Main module for start the application."""
from fastapi import Depends, FastAPI
from app.internal.routes import __routes__
from app.pkg.middlewares.x_auth_token import x_auth_token
from app.pkg.settings import settings

def create_app():
    """All settings for the application."""
    if settings.IS_PROD:
        configs = settings.API.CONFIGS
    else:
        configs = {}

    app = FastAPI(
        **configs,
        dependencies=[Depends(x_auth_token)],
    )
    __routes__.register_routes(app=app)
    return app
