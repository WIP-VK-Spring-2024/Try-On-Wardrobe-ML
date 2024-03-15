"""Models of model task object."""

from pydantic import PositiveInt
from pydantic.fields import Field
from pydantic import validator

from app.pkg.models.base import BaseModel as Model
from app.pkg.models.app.image_type import ImageType
from app.pkg.settings import settings

__all__ = [
    "CreateTaskCmd",
    "CreateTaskFileCmd",
    "CreateRespFileCmd",
]


class BaseModel(Model):
    """Base model for user."""


class ModelFields:
    """Model fields of user."""

    user_id: PositiveInt = Field(description="User id.", example=1)
    image_id: PositiveInt = Field(description="Image id.", example=1)
    image_type: ImageType = Field(
        description="Image type.",
        example=ImageType.CLOTH,
    )
    message: str = Field(description="Message.", example="Successfully uploaded.")
    file_name: str = Field(description="File name.")
    file_path: str = Field(
        description="file path.",
        default=str(settings.API_FILESYSTEM_FOLDER),
    )


class CreateTaskCmd(BaseModel):
    image_type: ImageType = ModelFields.image_type
    user_id: PositiveInt = ModelFields.user_id
    image_id: PositiveInt = ModelFields.image_id
    file_name: str = ModelFields.file_name


class CreateTaskFileCmd(CreateTaskCmd):
    file_path: str = ModelFields.file_path

class CreateRespFileCmd(CreateTaskFileCmd):
    pass


class ResponseMessage(BaseModel):
    message: str = ModelFields.message