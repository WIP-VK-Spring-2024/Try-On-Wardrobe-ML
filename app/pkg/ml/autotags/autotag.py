from typing import Dict, Union, List
import io

import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer

from app.pkg.ml.buffer_converters import BytesConverter

class AutoTagger:
    def __init__(self, device="cuda:0"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.bytes_converter = BytesConverter()


    def get_tags(self, input_data: Dict[str, Union[io.BytesIO, List[List[str]]]]) -> Dict[str, float]:
        """
        Gets probability for each tag
        
        Args:
            input_data - Dict[str, io.BytesIO] - dict, contained folowing structure:
                {
                "image":io.BytesIO,
                "tags":List[List[str]]
                }
        Returns:
            dict with probabilities
        
        Example for input tags:[["winter", "summer"], ["t-shirt","pants", "shoes"]]
        Will return {"winter":0.79, "summer":0.21, "t-shirt":0.1, "pants":0.15, "shoes":0.75}
        Multy list format need for normalizing values into probabilities
        """
        image = self.bytes_converter.bytes_to_image(input_data["image"])
        image_inputs = self.processor(images=image, return_tensors="pt")
        self._input_to_device(image_inputs)

        # image_inputs['pixel_values'] = image_inputs['pixel_values'].to(self.device)

        image_features = self.model.get_image_features(**image_inputs)

        text_tags_lists = input_data["tags"]
        text_features_list = []
        for text_tag_list in text_tags_lists:
            self.processor(text=text_tag_list, images=image, return_tensors="pt", padding=True)
            text_inputs = self.tokenizer(text_tag_list, padding=True, return_tensors="pt")
            self._input_to_device(text_inputs)

            text_features = self.model.get_text_features(**text_inputs)
            text_features_list.append(text_features)
        return image_features, text_features_list

    def _input_to_device(self, input_data:dict):
        """
        Converts input dict values to device
        """
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                input_data[key] = value.to(self.device)
                