"""Service for manage file working."""

import pathlib
from typing import Tuple
import uuid
from fastapi import UploadFile

from app.pkg.settings import settings

class FileService:

    def __init__(self) -> None:
        self.default_path = pathlib.Path(
            settings.API_FILESYSTEM_FOLDER,
            "try-on",
        ).absolute()
        if not self.default_path.exists():
            self.default_path.mkdir(exist_ok=True, parents=True)

    async def save_file(self, file: UploadFile, extension: str = 'png', file_id: uuid.uuid4 = None) -> Tuple[str, pathlib.Path]:
        file_name, file_path = self.get_file_path(file_id=file_id, extension=extension)
        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)

        return file_name, file_path
    
    def get_file_path(self, file_id: uuid.uuid4 = None, extension: str = 'png') -> Tuple[str, pathlib.Path]:
        file_id = file_id if file_id else uuid.uuid4()
        
        file_name = f'{file_id}.{extension}'
        file_path = pathlib.Path(self.default_path, file_name).absolute()

        return file_name, file_path
    
    def get_mock_file_path(self, file_id: uuid.uuid4 = None) -> Tuple[str, pathlib.Path]:
        file_id = file_id if file_id else "2a6cfbc7-09dd-4425-9227-0bdf7e70dd88.png"
        file_name = f'{file_id}.png'
        file_path = pathlib.Path(self.default_path, file_name).absolute()

        return file_name, file_path
