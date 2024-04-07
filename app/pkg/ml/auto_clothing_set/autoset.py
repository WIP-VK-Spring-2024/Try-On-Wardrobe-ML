from typing import Dict, Union, List
import io

import torch
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from torch.nn import Softmax
from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter



class LocalRecSys:
    """
    Recommends set of clothes (outfit)
    """
    def __init__(self, device="cuda:0"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.bytes_converter = BytesConverter()

    def forward(self,
                upper_clothes: List[Dict[str, io.BytesIO]],
                lower_clothes: List[Dict[str, io.BytesIO]],
                dresses_clothes: List[Dict[str, io.BytesIO]],
                user_photos:List[Dict[str, io.BytesIO]],
                prompt: str = None,
                top_n: int = 10,
               ) -> Dict[str, Dict[str, float]]:
        """
        Gets probability for each tag
        
        Args:
            upper_clothes: List[Dict[str, io.BytesIO]],
            lower_clothes: List[Dict[str, io.BytesIO]],
            dresses_clothes: List[Dict[str, io.BytesIO]],
            user_photos:List[Dict[str, io.BytesIO]] - photos of user to try_on in future.
                    Recommends to use one or less images
            prompt: str = None - extra prompt to search
        Returns:
            dict with sets of clothes
        
        Return example:
        [
            {
            "score":1,
            "clothes":[,]
            },
            {
            "score":0.93,
            "clothes":[,]
            },
        ]

        """
        pass
        # image = self.bytes_converter.bytes_to_image(input_data["image"])
        # tags = input_data['tags']
        # output_dict = self._get_tags(image, tags)
        # return output_dict

    @torch.inference_mode()
    def _get_image_embedding(self, images:List[Dict[str, Image]] ) -> Dict[str, float]:
        """
        Gets images embeddings
        
        Args:
            images:List[Dict[str, Image]] - images to get embeddings
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

            self.processor(text=text_tag_list, images=image, return_tensors="pt", padding=True)
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
                