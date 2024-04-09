from typing import Dict, Union, List
import io

import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from torch.nn import Softmax
from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter


def max_normalize(probs):
    return probs/probs.max().item()


class AutoTagger:
    def __init__(self, device="cuda:0"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.bytes_converter = BytesConverter()
        self.normalize = max_normalize#Softmax(dim=1)

    def forward(self, input_data: Dict[str, Union[io.BytesIO, Dict[str, List[str]]]]) -> Dict[str, Dict[str, float]]:
        """
        Gets probability for each tag
        
        Args:
            input_data - Dict[str, io.BytesIO] - dict, contained folowing structure:
                {
                "image":io.BytesIO,
                "tags":Dict[str, List[str]]
                }
        Returns:
            dict with probabilities
        
        Example for input tags:
        {
            "seasons": ["summer", "winter", "autumn", "spring"],
            "types": ["upper", "lower", "dresses", "shoes"]
        }

        Will return:
        {
            "seasons": {
                "summer": 0.1,
                "winter": 0.5,
                "autumn": 0.3,
                "spring": 0.1
            },
            "types": {...}
        }

        Multy list format need for normalizing values into probabilities
        """
        image = self.bytes_converter.bytes_to_image(input_data["image"])
        tags = input_data['tags']
        output_dict = self._get_tags(image, tags)
        return output_dict

    @torch.inference_mode()
    def _get_tags(self, image:Image, tags:Dict[str, List[str]]) -> Dict[str, float]:
        """
        Gets probability for each tag
        
        Args:
            image:Image - image to get tags
            tags:Dict[str, List[str]]] - tags
            
        Returns:
            dict with probabilities   
        """
        image_inputs = self.processor(images=image, return_tensors="pt")
        self._input_to_device(image_inputs)

        image_features = self.model.get_image_features(**image_inputs)

        result_probabilities = {}
        for tag_group, text_tag_list in tags.items():
            result_probabilities[tag_group] = {}

            # self.processor(text=text_tag_list, images=image, return_tensors="pt", padding=True)
            text_inputs = self.tokenizer(text_tag_list, padding=True, return_tensors="pt")
            self._input_to_device(text_inputs)

            text_features = self.model.get_text_features(**text_inputs)
            logits = image_features @ text_features.T
            probabilities = self.normalize(logits)
            probabilities = probabilities.flatten().tolist()

            for tag, prob in zip(text_tag_list, probabilities):
                r_prob = round(prob, 3)
                result_probabilities[tag_group][tag] = r_prob

        return result_probabilities

    def _input_to_device(self, input_data:dict):
        """
        Converts input dict values to device
        """
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                input_data[key] = value.to(self.device)
                