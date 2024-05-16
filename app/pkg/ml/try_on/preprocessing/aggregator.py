import io
from typing import Dict, Union

from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter

from app.pkg.ml.try_on.preprocessing.preprocessing import Resizer

from app.pkg.ml.try_on.preprocessing.pose import PoseEstimation
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor, BackgroundModels
from app.pkg.ml.try_on.preprocessing.human_parsing import HumanParsing 
from app.pkg.ml.try_on.preprocessing.dense_pose import DensePoseEstimation

class BaseProcessor:
    def __init__(self):
        self.bytes_converter = BytesConverter()

class ClothProcessor(BaseProcessor):
    """
    Class for processing clothes. Puts into worker
    """
    def __init__(self, model_type=BackgroundModels.BriaRMBG):
        super().__init__()
        self.model_background = ClothPreprocessor(model_type)        

    def consistent_forward(self, image_bytes:io.BytesIO) -> Dict[str, io.BytesIO]:
        """
        Processes cloth image
        Removes background from input image buffer

        Args:
            image_bytes - bytes of cloth image
        
        Returns:
            result - dict with Dict[str, io.BytesIO] format
        """
        image = self.bytes_converter.bytes_to_image(image_bytes)
        no_background_image = self.model_background(image)
        no_background_image_bytes = self.bytes_converter.image_to_bytes(no_background_image)

        result = {}

        result["cloth"] = no_background_image_bytes        
        
        return result


class HumanProcessor(BaseProcessor):
    """
    Class for processing clothes. Puts into worker
    """
    def __init__(self):
        super().__init__()
        self.resizer = Resizer(try_on_height=1024,
                               try_on_width=768)

        self.model_pose_estim = PoseEstimation()
        self.model_human_parsing = HumanParsing()
        self.model_dense_pose = DensePoseEstimation()
        

    def consistent_forward(self, image_bytes: io.BytesIO)->Dict[str, io.BytesIO]:
        """
        Starts processing of image with human.
        Args:
            image_bytes - human image in bytes format

        Returns:
            result - dict with Dict[str, io.BytesIO] format
        """
        image = self.bytes_converter.bytes_to_image(image_bytes)

        result = self.process(image)

        return result                

    def process(self, image:Image):
        result = {}

        result["image_human_orig"] = self.bytes_converter.image_to_bytes(image)

        human_resized_try_on = self.resizer(image, color=(255,255,255))
        result["image_human_try_on"] = self.bytes_converter.image_to_bytes(human_resized_try_on)

        human_resized_preproc = self.resizer.stretch_resize(human_resized_try_on, preproc=True)
        result["image_human_preproc"] = self.bytes_converter.image_to_bytes(human_resized_preproc)
    

        pose_out, keypoints_json_dict = self.model_pose_estim(human_resized_preproc)
        result["pose"] = self.bytes_converter.image_to_bytes(pose_out)
        result["keypoints_json"] = self.bytes_converter.json_to_bytes(keypoints_json_dict)

        parsed_human = self.model_human_parsing(human_resized_preproc)
        result["parse_orig"] = self.bytes_converter.image_to_bytes(parsed_human)

        dense_human_array = self.model_dense_pose(human_resized_preproc)
        dense_human = Image.fromarray(dense_human_array)
        result["dense_pose"] = self.bytes_converter.image_to_bytes(dense_human)

        return result

