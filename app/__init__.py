"""Main module for start the application."""
from fastapi import Depends, FastAPI
from app.internal.routes import __routes__
from app.pkg.middlewares.x_auth_token import x_auth_token

def create_app():
    """All settings for the application."""
    app = FastAPI(dependencies=[Depends(x_auth_token)])
    __routes__.register_routes(app=app)
    return app
