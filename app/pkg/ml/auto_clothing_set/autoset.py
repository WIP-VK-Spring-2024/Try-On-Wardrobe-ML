from typing import Dict, Union, List
import io
from copy import deepcopy

import torch
import numpy as np
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
from torch.nn import Softmax
from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.utils.device import get_device
from app.pkg.logger import get_logger

logger = get_logger(__name__)


def sum_normalize(array):
    if isinstance(array, list):
        array = np.array(array)
    return array/array.sum()


class LocalRecSys:
    """
    Recommends set of clothes (outfit)
    """
    def __init__(self,
                 return_cloth_fields = ["clothes_id",],
                 use_top_p=False,
                 calculate_tensors=False,
                 ):
        """
        Args:
            return_cloth_fields - fields, that must be in each returning cloth
            use_top_p - is need to use top_p algorithm
            calculate_tensors - is need to recalculate tensors data
        """
        self.device = get_device("cuda:1")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer =  AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_top_p = use_top_p
        self.calculate_tensors = calculate_tensors

        self.return_cloth_fields = return_cloth_fields
        self.bytes_converter = BytesConverter()

    def get_max_sets_amount(
                self,            
                upper_clothes: List[Dict[str, io.BytesIO]] = [],
                lower_clothes: List[Dict[str, io.BytesIO]] = [],
                dresses_clothes: List[Dict[str, io.BytesIO]] = [],
                outerwear_clothes: List[Dict[str, io.BytesIO]] = [],
                ):
        U = len(upper_clothes)
        L = len(lower_clothes)
        D = len(dresses_clothes)
        O = len(outerwear_clothes)
        return U*L*(O+1) + D


    def forward(self,
                upper_clothes: List[Dict[str, io.BytesIO]] = [],
                lower_clothes: List[Dict[str, io.BytesIO]] = [],
                dresses_clothes: List[Dict[str, io.BytesIO]] = [],
                outerwear_clothes: List[Dict[str, io.BytesIO]] = [],
                
                # user_photos:List[Dict[str, io.BytesIO]],
                prompt: str = None,
                sample_amount: int = 10,
               ) -> Dict[str, Dict[str, float]]:
        """
        Gets probability for each tag
        
        Args:
            upper_clothes: List[Dict[str, io.BytesIO]],
            lower_clothes: List[Dict[str, io.BytesIO]],
            dresses_clothes: List[Dict[str, io.BytesIO]],
            prompt: str = None - extra prompt to search
            sample_amount: int = 10 - samples of outfits  
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
        if prompt == '':
            prompt = None
        
        max_sets = self.get_max_sets_amount(
            upper_clothes,
            lower_clothes,
            dresses_clothes,
            outerwear_clothes,              
        )

        if max_sets == 0:
            logger.warn(f"Can't sample any sets. Not enough clothes")
            return []
        
        if sample_amount > max_sets:
            logger.warn("Sampling more than it could be."\
                        f"Sampling {sample_amount}, when max is {max_sets}")
        
        if self.calculate_tensors:
            upper_clothes = self.get_embs_for_clothes(upper_clothes,
                                                      return_list=False)
            lower_clothes = self.get_embs_for_clothes(lower_clothes,
                                                      return_list=False)
            dresses_clothes = self.get_embs_for_clothes(dresses_clothes,
                                                        return_list=False)
            outerwear_clothes = self.get_embs_for_clothes(outerwear_clothes,
                                                          return_list=False)
        else:
            upper_clothes = self.convert_tensors_from_list(upper_clothes)
            lower_clothes = self.convert_tensors_from_list(lower_clothes)
            dresses_clothes = self.convert_tensors_from_list(dresses_clothes,)
            outerwear_clothes = self.convert_tensors_from_list(outerwear_clothes,)

        if prompt is not None:
            prompt_features = self._get_text_embedding(prompt)
        else:
            prompt_features = None

        # перебор все возможных комбинаций одежды
        outfits = []
        for up_cloth in upper_clothes:
            for low_cloth in lower_clothes:
                for outer_wear_cloth in [*outerwear_clothes, None]:
                    outfit = {'clothes':[up_cloth, low_cloth]}
                    if outer_wear_cloth is not None:
                      outfit['clothes'].append(outer_wear_cloth)  
                    self._evaluate_outfit(outfit=outfit,
                        prompt_features=prompt_features)
                    outfits.append(outfit)
                
        for dress in dresses_clothes:         
            outfit = {'clothes':[dress]}
            self._evaluate_outfit(outfit=outfit,
                prompt_features=prompt_features)

            outfits.append(outfit)
        # for outfit in outfits:
        #     clothes_t = [cloth['tensor'].unsqueeze(0) for cloth in outfit['clothes']]
        #     outfit['tensor'] = torch.concat(clothes_t).mean(0)
        #     outfit['tensor'] = self.bytes_converter.torch_to_bytes(outfit['tensor'])
        # if 'tensor' not in self.return_cloth_fields:
        #     for cloth in [*upper_clothes, *lower_clothes, *dresses_clothes, *outerwear_clothes]:
        #         del cloth['tensor']

        sorted_outfits = sorted(outfits, key=lambda x: x['score'], reverse=True)

        # sample some outfits
        sampled_outfits = self.sample_outfit(sorted_outfits, sample_amount)
        sampled_outfits_cleared = self.stay_attributes(sampled_outfits)
        return sampled_outfits_cleared

    def stay_attributes(self, outfits):
        outfits_cleared_fields = []
        for outfit in outfits:
            outfit_clothes = []
            for cloth in outfit['clothes']:
                
                new_cloth = {}
                for parameter in self.return_cloth_fields:
                    new_cloth[parameter] = cloth[parameter]
                outfit_clothes.append(new_cloth)
            outfits_cleared_fields.append({'clothes':outfit_clothes,})
                                        #    'tensor':outfit['tensor']})
        return outfits_cleared_fields             

    def _evaluate_outfit(self, outfit, prompt_features):
        score_list = [cloth['tensor'] for cloth in outfit['clothes']]

        if prompt_features is not None:
            prompt_correlations = [(prompt_features@score).item() for score in score_list]
        else:
            prompt_correlations=[0]

        if len(outfit['clothes']) > 1:
            clothes_score = self.dot_product(*score_list).item()/len(outfit['clothes'])
        else:
            clothes_score = 0

        outfit['score'] = 25 * np.mean(prompt_correlations) + clothes_score
        outfit['clothes_score'] = clothes_score
        outfit['prompt_corr'] = np.mean(prompt_correlations)

    def convert_tensors_from_list(self, clothes:List[Dict[str, list]]):
        converted_clothes = []
        for cloth in clothes:
            cloth_copy = deepcopy(cloth)
            tensor = torch.tensor(cloth['tensor'])
            cloth_copy['tensor'] = tensor
            # print(self.bytes_converter.bytes_to_torch(tensor))
            converted_clothes.append(cloth_copy)           
        return converted_clothes

    def convert_tensors_from_bytes(self,
                                   clothes: List[Dict[str, io.BytesIO]]):
        converted_clothes = []
        for cloth in clothes:
            cloth_copy = deepcopy(cloth)
            tensor = cloth['tensor']
            cloth_copy['tensor'] = self.bytes_converter.bytes_to_torch(tensor)
            # print(self.bytes_converter.bytes_to_torch(tensor))
            converted_clothes.append(cloth_copy)
            
        return converted_clothes

    def get_embs_for_clothes(self,
                             clothes: List[Dict[str, io.BytesIO]],
                             return_list=True):
        clothes = self.prepare_clothes(clothes)
        pil_clothes = [cloth['cloth'] for cloth in clothes]
        image_features = self._get_images_embedding(pil_clothes).cpu()
        for cloth, tensor in zip(clothes, image_features):
            if return_list:
                cloth['tensor'] = tensor.tolist()
            else:
                cloth['tensor'] = tensor
        return clothes


    def prepare_clothes(self, clothes: list):
        new_clothes = []

        if len(clothes) != 0:
            clothes_params = clothes[0].keys()

            for cloth in clothes:
                new_cloth = {}
                cloth_no_background = self.bytes_converter.bytes_to_image(cloth['cloth'])
                white_background_cloth = ClothPreprocessor.replace_background_RGBA(
                                                            cloth_no_background,
                                                            color=(255,255,255)
                                                            )
                for cloth_param in clothes_params:
                    new_cloth[cloth_param] = cloth[cloth_param]

                new_cloth['cloth'] = white_background_cloth
                new_clothes.append(new_cloth)
        return new_clothes


    @torch.inference_mode()
    def _get_text_embedding(self, text:str ) -> torch.tensor:
        """
        Gets images embeddings
        
        Args:
            images:List[Dict[str, Image]] - images to get embeddings
            
        Returns:
            original dict but with embeddings
        """

        if len(text) == 0:
            raise ValueError("Got empty text")

        text_inputs = self.tokenizer(text, padding=True, return_tensors="pt")
        self._input_to_device(text_inputs)

        text_features = self.model.get_text_features(**text_inputs)

        return text_features


    @torch.inference_mode()
    def _get_images_embedding(self, images:List[Image.Image] ) \
                                ->List[Dict[str, Union[Image.Image, torch.tensor]]]:
        """
        Gets images embeddings
        
        Args:
            images:List[Dict[str, Image]] - images to get embeddings
            
        Returns:
            original dict but with embeddings
        """

        if len(images) == 0:
            return []

        image_inputs = self.processor(images=images, return_tensors="pt")
        self._input_to_device(image_inputs)

        image_features = self.model.get_image_features(**image_inputs)

        return image_features

    def sample_outfit(self, outfits, sample_amount):

        scores = torch.tensor([outfit['score']/30 for outfit in outfits])
        normalized_scores = self.softmax(scores)

        if self.use_top_p:
            top_p_scores = None
            top_p_index = None
            top_p = 0.9
            for i, score in enumerate(normalized_scores):
                if normalized_scores[:i].sum() > top_p:
                    top_p_scores = normalized_scores[:i]
                    top_p_index = i
                    break
            else:
                logger.info("Got case top_p not reached")
                top_p_scores = normalized_scores

            new_scores = sum_normalize(top_p_scores)
        else:
            new_scores = normalized_scores

        indexes = np.arange(len(new_scores))
        sampled_indexes = np.random.choice(indexes,(sample_amount,), p=new_scores ,replace=False)
        filtered_outfit = [outfits[i] for i in sampled_indexes]
        return filtered_outfit

    def _input_to_device(self, input_data:dict):
        """
        Converts input dict values to device
        """
        for key, value in input_data.items():
            if isinstance(value, torch.Tensor):
                input_data[key] = value.to(self.device)

    @staticmethod
    def dot_product(*args: List[torch.tensor]):

        num = 1
        for i in args:
            if i is not None:
                num *= i.flatten()
        return num.sum()
