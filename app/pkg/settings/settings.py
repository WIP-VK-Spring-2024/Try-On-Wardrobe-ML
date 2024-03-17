"""Module for load settings form `.env`"""

import pathlib
from typing import Optional
import urllib.parse
from functools import lru_cache

from dotenv import find_dotenv
from pydantic import PostgresDsn, validator, root_validator, AmqpDsn
from pydantic import AnyUrl, BaseSettings, SecretStr, PositiveInt

from app.pkg.models.core.logger import LoggerLevel


__all__ = ["Settings"]


class _Settings(BaseSettings):

    class Config:
        """Configuration of settings."""

        #: str: env file encoding.
        env_file_encoding = "utf-8"
        #: str: allow custom fields in model.
        arbitrary_types_allowed = True
        #: bool: case-sensitive for env variables.
        case_sensitive = True
        #: str: delimiter for nested env variables.
        env_nested_delimiter = "__"


class Resource(_Settings):
    """Resource base settings."""

    #: str: Resource host.
    HOST: str
    #: PositiveInt: positive int (x > 0) port of Resource.
    PORT: PositiveInt

    #: str: Resource. (DSN)
    #  Builds in `root_validator` method.
    DSN: Optional[str] = None

    def build_dsn(cls, values: dict):  # pylint: disable=no-self-argument
        raise NotImplementedError


class Postgresql(Resource):
    """Postgresql settings."""

    #: str: Postgresql host.
    HOST: str = "localhost"
    #: PositiveInt: positive int (x > 0) port of postgresql.
    PORT: PositiveInt = 5432
    #: str: Postgresql user.
    USER: str = "postgres"
    #: SecretStr: Postgresql password.
    PASSWORD: SecretStr = SecretStr("postgres")
    #: str: Postgresql database name.
    DATABASE_NAME: str = "postgres"

    #: str: Concatenation all settings for postgresql in one string. (DSN)
    #  Builds in `root_validator` method.
    DSN: Optional[str] = None

    @root_validator(pre=True)
    def build_dsn(cls, values: dict):  # pylint: disable=no-self-argument
        values["DSN"] = PostgresDsn.build(
            scheme="postgresql",
            user=f"{values.get('USER')}",
            password=f"{urllib.parse.quote_plus(values.get('PASSWORD'))}",
            host=f"{values.get('HOST')}",
            port=f"{values.get('PORT')}",
            path=f"/{values.get('DATABASE_NAME')}",
        )
        return values


class RabbitMQ(Resource):
    """RabbitMQ settings."""

    #: str: Resource host.
    HOST: str = "localhost"
    #: PositiveInt: positive int (x > 0) port of Resource.
    PORT: PositiveInt = 5672
    #: str: Resource user.
    USER: str = "user"
    #: SecretStr: Resource password.
    PASSWORD: SecretStr = "secret"

    @root_validator(pre=True)
    def build_dsn(cls, values: dict):  # pylint: disable=no-self-argument
        values["DSN"] = AmqpDsn.build(
            scheme="amqp",
            user=f"{values.get('USER')}",
            password=f"{urllib.parse.quote_plus(values.get('PASSWORD'))}",
            host=f"{values.get('HOST')}",
            port=f"{values.get('PORT')}",
        )
        return values


class Resources(_Settings):
    POSTGRES: Postgresql
    RABBITMQ: RabbitMQ


class Client(_Settings):
    URL: AnyUrl
    TOKEN: SecretStr


class Clients(_Settings):
    WARDROBE_API: Client


class APIServer(_Settings):
    """API settings."""
    INSTANCE_APP_NAME: str = 'api'
    #: PositiveInt: positive int (x > 0) port of API.
    PORT: PositiveInt = 8000
    #: str: Trusted APIs from
    TRUSTED: str = "*"

    # --- SECURITY SETTINGS ---
    #: SecretStr: X-ACCESS-TOKEN for access to API.
    X_ACCESS_TOKEN: SecretStr = "secret"


class Logging(_Settings):
    """Logging settings."""

    #: StrictStr: Level of logging which outs in std
    LEVEL: LoggerLevel = LoggerLevel.DEBUG
    #: pathlib.Path: Path of saving logs on local storage.
    FOLDER_PATH: pathlib.Path = pathlib.Path("./volume/logs")

    @validator("FOLDER_PATH")
    def __create_dir_if_not_exist(  # pylint: disable=unused-private-member, no-self-argument
        cls,
        v: pathlib.Path,
    ):
        """Create directory if not exist."""

        if not v.exists():
            v.mkdir(exist_ok=True, parents=True)
        return v


class ML(_Settings):
    """ML settings."""

    #: pathlib.Path: Path of models weights.
    WEIGHTS_PATH: pathlib.Path = pathlib.Path("./app/pkg/ml/weights")


class Settings(_Settings):
    """APP settings."""

    API: APIServer

    RESOURCES: Resources

    CLIENTS: Clients

    LOGGER: Logging

    ML: ML

    API_FILESYSTEM_FOLDER: pathlib.Path = pathlib.Path("./volume/data")

    class Config:
        env_file = ".env"


@lru_cache
def get_settings(env_file: str = ".env") -> Settings:
    """Create settings instance."""

    return Settings(_env_file=find_dotenv(env_file))