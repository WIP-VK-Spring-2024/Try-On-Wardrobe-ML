from enum import Enum

FILE_TYPES = Enum("json", "image", "npz")

class BaseFileManager:
    """
    Base class for managing files in ml part
    """
    def __init__(self):
        pass

    def save(self, variable, file_type:FILE_TYPES, path:str):
        """
        variable - object to save
        file_type - type of object. Possible values: json, image, npz
        path - path to save
        """
        pass


    def delete(self, path):
        """
        path - path to remove file from
        """
        pass

