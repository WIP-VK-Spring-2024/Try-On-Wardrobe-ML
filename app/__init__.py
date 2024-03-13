"""Main module for start the application."""
from fastapi import FastAPI
from app.model import model_router


def create_app():
    """All settings for the application must be here."""
    app = FastAPI()
    app.include_router(model_router, prefix="/data", tags=["data"])

    return app
