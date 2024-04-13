from typing import List, Dict, Union
import io
from copy import deepcopy

import torch

from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.try_on.postprocessing.fix_face import FaceFixer
from app.pkg.ml.buffer_converters import BytesConverter
from app.pkg.models.app.image_category import ImageCategory

from app.pkg.logger import get_logger

logger = get_logger(__name__)

class LadyVtonAggregator:

    def __init__(self):
        self.preprocessor = LadyVtonInputPreprocessor()
        self.model = LadyVton()
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
                "category":ImageCategory, # one of ['dresses', 'upper_body','lower_body']
                }        

        """

        #self.prepare_input_data(input_data)
        self.prepare_cloth(input_data)
        self.prepare_human(input_data)


        # result_image = Image.new("RGB", input_data["image_human_orig"].size, )

        result_image = self.model.forward(input_data)
        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=input_data["image_human_orig"],
            result_image=result_image)
        return self.bytes_converter.image_to_bytes(fixed_face_image)


    @torch.inference_mode()
    def try_on_set(self, human: Dict[str, io.BytesIO],
                   clothes: List[Dict[str, Union[io.BytesIO, ImageCategory]]],
                   ) -> io.BytesIO:
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
        logger.info("Starting try on outfit")

        # convert bytearray into pil image
        self.prepare_human(human, to_preprocessor=False)
        #result_image = human["image_human_orig"]
        result_image = human["image_human_orig"]

        for cloth in clothes:
            self.prepare_cloth(cloth)

        # find a cloth with lower body
        for cloth in clothes:
           # assert isinstance(cloth, ImageCategory)
            if cloth["category"] == ImageCategory.UPPER_BODY:
                logger.info("[TryOnSet] Found upper body cloth")
                upper_human = deepcopy(human)
                upper_human['category'] = ImageCategory.UPPER_BODY
                self.preprocessor.prepare_human(upper_human)

                input_data = self.get_try_on_data(human=upper_human, cloth=cloth)
                result_image = self.model.forward(input_data)
                break
        else:
            logger.warn(f"[TryOnSet] Not found upper body cloth")
    
        for cloth in clothes:
            if cloth["category"] == ImageCategory.LOWER_BODY:
                logger.info("[TryOnSet] Found lower body cloth")
                lower_human = deepcopy(human)
                lower_human['category'] = ImageCategory.LOWER_BODY
                lower_human['image_human_orig'] = result_image # making the input, output of previous step
                self.preprocessor.prepare_human(lower_human)
                input_data = self.get_try_on_data(human=lower_human, cloth=cloth)
                result_image = self.model.forward(input_data)
                human['image'] = result_image
                break
        else:
            logger.warn(f"[TryOnSet] Not found lower body cloth")


        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=human["image_human_orig"],
            result_image=result_image)

        return self.bytes_converter.image_to_bytes(fixed_face_image)


    def get_try_on_data(self, human: dict, cloth:dict):
        input_data = {}
        input_data.update(human)
        input_data.update(cloth)
        return input_data


    def prepare_human(self, human: dict, to_pil=True, to_preprocessor=True):
        """
        Converts data types from byte array
        Args:
            human - dict with human prepr data
            to_pil - bool, true if need to convert from bytearray
            to_preprocessor - bool, true if need to use
                preprocessor.prepare_human()
        """
        if to_pil:
            human["image_human_orig"] = self.bytes_converter.bytes_to_image(

            human["image_human_orig"])

            human["parse_orig"] = self.bytes_converter.bytes_to_image(
                human["parse_orig"]
            )

            human["keypoints_json"] = self.bytes_converter.bytes_to_json(
                human["keypoints_json"]
            )

        if to_preprocessor:
            self.preprocessor.prepare_human(human)


    def prepare_cloth(self, cloth):
        """
        Converts data types from byte array
        """
        cloth_rgba = self.bytes_converter.bytes_to_image(
            cloth["cloth"])
        cloth['cloth'] = ClothPreprocessor.replace_background_RGBA(cloth_rgba,
                                                                        color=(255,255,255))
        self.preprocessor.prepare_cloth(cloth)


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
