
from typing import Dict, Union
import io

from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.buffer_converters import BytesConverter


class LadyVtonAggregator:

    def __init__(self):
        self.preprocessor = LadyVtonInputPreprocessor()
        self.model = LadyVton()
        self.bytes_converter = BytesConverter()

    def __call__(self, input_data: Dict[str, Union[io.BytesIO, str]]) -> Dict[str, io.BytesIO]:
        """
        Starts try on process
        
        Args:
            input_data - Dict[str, io.BytesIO] - dict, contained folowing structure:
                {
                "image_human_orig":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "pose":io.BytesIO,  # - human pose
                "keypoints":io.BytesIO # human keypoints json
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":str, # one of ['dresses', 'upper_body','lower_body']

                }
        
        """
        
        #TODO: Convert key to appropriate and its values. Keys must be same as in lady_vton_prepr. Values must be converted to pil
        input_data["image_human_orig"] = self.bytes_converter.bytes_to_image(
            input_data["image_human_orig"]) 
        
        cloth_rgba = self.bytes_converter.bytes_to_image(
            input_data["cloth"])
        input_data['cloth'] = ClothPreprocessor.replace_background_RGBA(cloth_rgba, color=255)
        # TODO: convert into rgb with white background

        input_data["parse_orig"] = self.bytes_converter.bytes_to_image(
            input_data["parse_orig"]
        )
        
        # input_data["keypoints_json"] = self.bytes_converter.bytes_to_image(
        #     input_data["keypoints"]
        # )

        input_data["keypoints_json"] = self.bytes_converter.bytes_to_json(
            input_data["keypoints_json"]
        )

        self.preprocessor(input_data)

        # TODO: implement ladyvton model here
        result_image = Image.new("RGB", input_data["image_human_orig"].size, )

        return self.bytes_converter.image_to_bytes(result_image)

if __name__ == '__main__':

    bytes_converter = BytesConverter()

    from PIL import Image
    import json

    lva = LadyVtonAggregator()
    input_data = {}

    cloth_path = "/usr/src/app/volume/data/no_background/cloth_prepr_ex.png"
    json_path = "/usr/src/app/volume/data/pose/keypoints1.json"
    parsed_human_path = "/usr/src/app/volume/data/parsed/parsed_human_aggregator.png"
    human_path = "/usr/src/app/data/example/human-dc.jpg"

    input_data["cloth"] = bytes_converter.image_to_bytes(
        Image.open(cloth_path))
    
    # a = ClothPreprocessor.replace_background_RGBA(Image.open(cloth_path), color=(255,255,255))
    # a.save("/usr/src/app/1.jpg")

    input_data["parse_orig"] = bytes_converter.image_to_bytes(
        Image.open(parsed_human_path))
    input_data["image_human_orig"] = bytes_converter.image_to_bytes(
        Image.open(human_path))
    
    with open(json_path, 'r') as f:
        json_keypoints = json.load(f)    
        input_data["keypoints_json"] = bytes_converter.json_to_bytes(json_keypoints)
    
    input_data['category'] = 'upper_body'

    result_bytes = lva(input_data)
    result_image = bytes_converter.bytes_to_image(result_bytes)
   # result_image.save("/usr/src/app/1.jpg")