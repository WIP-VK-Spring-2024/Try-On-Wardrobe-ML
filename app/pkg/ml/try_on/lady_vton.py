from typing import Dict, Union
import json
import io

from PIL import Image
import torch

from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.try_on.postprocessing.fix_face import FaceFixer
from app.pkg.ml.buffer_converters import BytesConverter


class LadyVtonAggregator:

    def __init__(self):
        self.preprocessor = LadyVtonInputPreprocessor()
        self.model = LadyVton(num_inference_steps=20)
        self.face_fix_model = FaceFixer()
        self.bytes_converter = BytesConverter()

    @torch.inference_mode()
    def __call__(self, input_data: Dict[str, Union[io.BytesIO, str]]) -> Dict[str, io.BytesIO]:
        """
        Starts try on process
        
        Args:
            input_data - Dict[str, io.BytesIO] - dict, contained folowing structure:
                {
                "image_human_orig":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "keypoints_json":io.BytesIO # human keypoints json
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":str, # one of ['dresses', 'upper_body','lower_body']

                }
        
        """
        
        input_data["image_human_orig"] = self.bytes_converter.bytes_to_image(
            input_data["image_human_orig"]) 
        
        cloth_rgba = self.bytes_converter.bytes_to_image(
            input_data["cloth"])
        input_data['cloth'] = ClothPreprocessor.replace_background_RGBA(cloth_rgba, color=(255,255,255))

        input_data["parse_orig"] = self.bytes_converter.bytes_to_image(
            input_data["parse_orig"]
        )
        
        input_data["keypoints_json"] = self.bytes_converter.bytes_to_json(
            input_data["keypoints_json"]
        )

        self.preprocessor(input_data)

        # result_image = Image.new("RGB", input_data["image_human_orig"].size, )
        
        result_image = self.model.forward(input_data)
        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=input_data["image_human_orig"],
            result_image=result_image)
        return self.bytes_converter.image_to_bytes(fixed_face_image)
