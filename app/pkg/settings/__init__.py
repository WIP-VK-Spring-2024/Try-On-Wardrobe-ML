"""Module of settings."""

from app.pkg.settings.settings import Settings, get_settings

__all__ = ["settings"]

settings: Settings = get_settings()
