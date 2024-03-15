from enum import Enum

__all__ = ["BaseEnum"]


class BaseEnum(Enum):
    """Base ENUM model."""

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.value)
