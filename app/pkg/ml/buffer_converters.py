import io
import json

import numpy as np
import torch
from PIL import Image


class BytesConverter:

    def __init__(self):
        pass

    def np_array_to_buffer(self, np_array):
        buffer = io.BytesIO()

        # Сохраняем массив в буфер в формате NPY
        np.save(buffer, np_array)

        # Получаем содержимое буфера в виде байтов
        buffer.seek(0)
        data = buffer.read()
        return data

    def buffer_to_np_array(self, buffer: io.BytesIO):

        # Загружаем массив NumPy из буфера
        arr = np.load(buffer)
        return arr

    def json_to_bytes(self, json_dict):
        return json.dumps(json_dict) # Тут может возникнуть проблема с взаимодействием с s3

    def bytes_to_json(self, bytes: io.BytesIO):
        # Преобразуем JSON-строку обратно в словарь
        return json.loads(bytes)

    def torch_to_buffer(self, tensor):
        buffer = io.BytesIO()

        # Сохраняем тензор в буфер
        torch.save(tensor, buffer)

        # Получаем содержимое буфера в виде байтов
        buffer.seek(0)
        tensor_data = buffer.read()
        return tensor_data

    def buffer_to_torch(self, buffer: io.BytesIO):
        # Загружаем тензор из буфера
        loaded_tensor = torch.load(buffer)
        return loaded_tensor

    def bytes_to_image(self, buffer: io.BytesIO):
        """
        Returns PIL image 
        """
        pil_image = Image.open(buffer)
        return pil_image


    def image_to_bytes(self, image):
        """
        image - PIL Image
        """
        # Создаем буфер памяти
        buffer = io.BytesIO()

        # Сохраняем изображение в буфер в формате PNG
        image.save(buffer, format='PNG')

        # Получаем содержимое буфера в виде байтов
        return buffer
        # buffer.seek(0)
        # image_data = buffer.read()
        # return image_data
