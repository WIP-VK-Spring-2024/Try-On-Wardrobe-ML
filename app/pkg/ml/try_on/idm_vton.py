from typing import List, Dict, Union
import io
from copy import deepcopy

import torch




from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
# from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
# from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor


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
                "cloth_desc":str # description of cloth. Mainly cloth subcategory
                "category":ImageCategory, # one of ['dresses', 'upper_body','lower_body']
                }
        """

        self.prepare_cloth(input_data)
        self.prepare_human(input_data)

        result_image = self.model.forward(input_data)
        fixed_face_image = self.face_fix_model.fix_face(
            orig_image=input_data["image_human_orig"],
            result_image=result_image)
        return self.bytes_converter.image_to_bytes(fixed_face_image)

    @torch.inference_mode()
    def batch_try_on(self,
                     human: Dict[str, io.BytesIO],
                     clothes: List[Dict[str, Union[io.BytesIO, ImageCategory]]]) -> io.BytesIO:
        """
        Starts try on process
        
        Args:
            input_data - Dict[str, Union[io.BytesIO, ImageCategory]] - dict, contained folowing structure:
                {
                "image_human_orig":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "keypoints_json":io.BytesIO, # human keypoints json
                }
            clothes - List[Dict[str, Union[io.BytesIO, ImageCategory]]] with format:
                {
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":ImageCategory, # one of ['dresses', 'upper_body','lower_body']
                }        

        """

        self.prepare_human(human, to_preprocessor=False)

        input_data = {
            'image':[],
            'inpaint_mask':[],
            'pose_map':[],
            'category':[],
            'cloth':[],
            'im_mask':[],
            'image_human_orig':[],
        }

        for cloth in clothes:
            self.prepare_cloth(cloth)

            human_per_cloth = deepcopy(human)
            human_per_cloth['category'] = cloth["category"]
            self.preprocessor.prepare_human(human_per_cloth)

            input_data['cloth'].append(cloth['cloth'])
            input_data['image'].append(human_per_cloth['image'])
            input_data['inpaint_mask'].append(human_per_cloth['inpaint_mask'])
            input_data['pose_map'].append(human_per_cloth['pose_map'])
            input_data['category'].append(human_per_cloth['category'])
            input_data['im_mask'].append(human_per_cloth['im_mask'])
            input_data['image_human_orig'].append(human_per_cloth['image_human_orig'])

        #input_data['cloth'] = clothes

        result_images = self.model.forward(input_data, single_cloth=False)

        fixed_face_results = []
        for i, result_image in enumerate(result_images):
            fixed_face_image = self.face_fix_model.fix_face(
                orig_image=input_data["image_human_orig"][i],
                result_image=result_image)
            fixed_im_bytes = self.bytes_converter.image_to_bytes(fixed_face_image)
            fixed_face_results.append(fixed_im_bytes)
        return fixed_face_results


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
                "cloth":io.BytesIO # cloth (without background) image bytes,
                "cloth_desc":str, # cloth description. Generally, cloth subcategory
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
            if cloth["category"] == ImageCategory.UPPER_BODY\
                or cloth["category"] == ImageCategory.DRESSES:

                logger.info("[TryOnSet] Found upper|dress body cloth")
                upper_human = deepcopy(human)
                upper_human['category'] = cloth["category"]
                self.preprocessor.prepare_human(upper_human)

                input_data = self.get_try_on_data(human=upper_human, cloth=cloth)
                result_image = self.model.forward(input_data)
                break
        else:
            logger.warn("[TryOnSet] Not found upper body cloth")

        for cloth in clothes:
            if cloth["category"] == ImageCategory.LOWER_BODY:
                logger.info("[TryOnSet] Found lower body cloth")
                lower_human = deepcopy(human)
                lower_human['category'] = cloth["category"]
                # making the input, output of previous step
                lower_human['image_human_orig'] = result_image 
                self.preprocessor.prepare_human(lower_human)
                input_data = self.get_try_on_data(human=lower_human, cloth=cloth)
                result_image = self.model.forward(input_data)
                human['image'] = result_image
                break
        else:
            logger.warn("[TryOnSet] Not found lower body cloth")


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
        if "cloth_desc" not in cloth.keys():
            cloth["cloth_desc"] = None

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
