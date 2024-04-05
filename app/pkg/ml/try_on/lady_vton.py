from typing import List, Dict, Union
import json
import io

from PIL import Image
import torch

from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.try_on.postprocessing.fix_face import FaceFixer
from app.pkg.ml.buffer_converters import BytesConverter
from app.pkg.models.app.image_category import ImageCategory


class LadyVtonAggregator:

    def __init__(self):
        self.preprocessor = LadyVtonInputPreprocessor()
        self.model = LadyVton(num_inference_steps=20)
        self.face_fix_model = FaceFixer()
        self.bytes_converter = BytesConverter()


    @torch.inference_mode()
    def __call__(self, input_data: Dict[str, Union[io.BytesIO, ImageCategory]]) -> io.BytesIO:
        """
        Starts try on process
        
        Args:
            input_data - Dict[str, Union[io.BytesIO, ImageCategory]] - dict, contained folowing structure:
                {
                "image_human_orig":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "keypoints_json":io.BytesIO # human keypoints json
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":str, # one of ['dresses', 'upper_body','lower_body']
                }        

        """

        self.prepare_input_data(input_data)
        # result_image = Image.new("RGB", input_data["image_human_orig"].size, )

        result_image = self.model.forward(input_data)
        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=input_data["image_human_orig"],
            result_image=result_image)
        return self.bytes_converter.image_to_bytes(fixed_face_image)


    @torch.inference_mode()
    def try_on_set(self, human: Dict[str, io.BytesIO],
                   clothes: List[Dict[str, Union[io.BytesIO, ImageCategory]]],
                   ) -> Dict[str, io.BytesIO]:
        """
        Try on set function

        Args:
            human - Dict[str, io.BytesIO] with format:
                {
                "image_human_orig":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "keypoints_json":io.BytesIO # human keypoints json
                }
            clothes - List[Dict[str, Union[io.BytesIO, ImageCategory]]] with format:
                {
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":ImageCategory, # one of ['dresses', 'upper_body','lower_body']
                }                        
        """
        self.prepare_human(human)
        for cloth in clothes:
            self.prepare_cloth(cloth)
                # result_image = self.model.forward(cloth)
                # fixed_face_image = self.face_fix_model.fix_face(
                #     orig_image=input_data["image_human_orig"],
                #     result_image=result_image)

        for cloth in clothes:
            if cloth["category"] == ImageCategory.LOWER_BODY:
                pass

        for cloth in clothes:
            if cloth["category"] == ImageCategory.UPPER_BODY:
                pass


        result_image = self.model.forward(input_data)
        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=input_data["image_human_orig"],
            result_image=result_image)

        return self.bytes_converter.image_to_bytes(fixed_face_image)


    def prepare_human(self, human):
        human["image_human_orig"] = self.bytes_converter.bytes_to_image(
        human["image_human_orig"])

        human["parse_orig"] = self.bytes_converter.bytes_to_image(
            human["parse_orig"]
        )

        human["keypoints_json"] = self.bytes_converter.bytes_to_json(
            human["keypoints_json"]
        )


    def prepare_cloth(self, cloth):
        cloth_rgba = self.bytes_converter.bytes_to_image(
            cloth["cloth"])
        cloth['cloth'] = ClothPreprocessor.replace_background_RGBA(cloth_rgba,
                                                                        color=(255,255,255))

    def prepare_input_data(self, input_data):
        """
        Converts input data objects to pil
        """
        input_data["image_human_orig"] = self.bytes_converter.bytes_to_image(
        input_data["image_human_orig"])

        cloth_rgba = self.bytes_converter.bytes_to_image(
            input_data["cloth"])
        input_data['cloth'] = ClothPreprocessor.replace_background_RGBA(cloth_rgba,
                                                                        color=(255,255,255))

        input_data["parse_orig"] = self.bytes_converter.bytes_to_image(
            input_data["parse_orig"]
        )

        input_data["keypoints_json"] = self.bytes_converter.bytes_to_json(
            input_data["keypoints_json"]
        )

        self.preprocessor(input_data)

        #return input_data
