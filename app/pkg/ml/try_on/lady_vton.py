
from typing import Dict, Union
import io

from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor
from app.pkg.ml.try_on.ladi_vton.lady_vton import LadyVton
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
                "image":io.BytesIO,  # - image with human
                "parsed_human":io.BytesIO,  # - image with parsed human 
                "pose":io.BytesIO,  # - human pose
                "keypoints":io.BytesIO # human keypoints json
                "cloth":io.BytesIO # cloth (without background) image bytes
                "category":str, # one of ['dresses', 'upper_body','lower_body']

                }
        
        """
        
        #TODO: Convert key to appropriate and its values. Keys must be same as in lady_vton_prepr. Values must be converted to pil
        input_data["image_human_orig"] = self.bytes_converter.bytes_to_image(
            input_data["image"]) 
        
        input_data["cloth"] = self.bytes_converter.bytes_to_image(
            input_data["cloth"])

        input_data["parse_orig"] = self.bytes_converter.bytes_to_image(
            input_data["parsed_human"]
        )
        
        input_data["keypoints_json"] = self.bytes_converter.bytes_to_image(
            input_data["keypoints"]
        )

        input_data["keypoints_json"] = self.bytes_converter.bytes_to_json(
            input_data["keypoints"]
        )
        
        
        
        self.preprocessor(input_data)

if __name__ == '__main__':
    lva = LadyVtonAggregator()
    cloth_path = "/usr/src/app/volume/data/no_background/cloth_prepr_ex.png"
    json_path = "/usr/src/app/volume/data/pose/keypoints1.json"
    parsed_human_path = "/usr/src/app/volume/data/parsed/parsed_human_aggregator.png"
    human_path = "/usr/src/app/data/example/human-dc.jpg"
