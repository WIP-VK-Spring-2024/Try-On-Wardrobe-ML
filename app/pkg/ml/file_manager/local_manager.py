import os
import json

import cv2
import torch

from app.pkg.ml.file_manager.base_manager import BaseFileManager, FILE_TYPES


class LocalFileManager(BaseFileManager):
    """
    File manager for saving/deleting files locally
    """
    def __init__(self,):
        super(LocalFileManager).__init__()

    def save(self, variable, file_type:FILE_TYPES, path:str):
        """
        variable - object to save
        file_type - type of object. Possible values: json, image, npz
        path - path to save
        """
        if file_type == FILE_TYPES.image:
            cv2.imwrite(path, file)

        elif file_type == FILE_TYPES.json:
            # saving json files, for example keypoints
            with open(path, 'w') as fin:
                fin.write(json.dumps(file))

        elif file_type == FILE_TYPES.npz:
            # saving exra data for 
            with open(path, "wb") as hFile:
                torch.save(file, hFile)

        else:
            raise TypeError("Unknown file type")

    def delete(self, path):
        os.remove(path)


if __name__ == '__main__':
    LocalFileManager()
