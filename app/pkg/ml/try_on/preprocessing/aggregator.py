import io
from typing import Dict, Union

from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter

from app.pkg.ml.try_on.preprocessing.preprocessing import Resizer

from app.pkg.ml.try_on.preprocessing.pose import PoseEstimation
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor
from app.pkg.ml.try_on.preprocessing.human_parsing import HumanParsing 


class BaseProcessor:
    def __init__(self):
        self.bytes_converter = BytesConverter()


class ClothProcessor(BaseProcessor):
    """
    Class for processing clothes. Puts into worker
    """
    def __init__(self):
        super().__init__()
        self.model_background = ClothPreprocessor()        

    def consistent_forward(self, image_bytes:io.BytesIO)->Dict[str, io.BytesIO]:
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
        self.resizer = Resizer()

        self.model_pose_estim = PoseEstimation()
        self.model_human_parsing = HumanParsing()

    def consistent_forward(self, image_bytes: io.BytesIO)->Dict[str, io.BytesIO]:
        """
        Starts processing of image with human.
        Args:
            image_bytes - human image in bytes format

        Returns:
            result - dict with Dict[str, io.BytesIO] format
        """
        image = self.bytes_converter.bytes_to_image(image_bytes)

        human_resized = self.resizer(image)

        result = {}
        pose_out, keypoints_json_dict = self.model_pose_estim(human_resized)
        result["pose"] = self.bytes_converter.image_to_bytes(pose_out)
        result["keypoints_json"] = self.bytes_converter.json_to_bytes(keypoints_json_dict)

        parsed_human = self.model_human_parsing(human_resized)
        result["parse_orig"] = self.bytes_converter.image_to_bytes(parsed_human)

        return result                


if __name__ == '__main__':
    bc = BytesConverter()
    
    cp = ClothProcessor()
    cloth_path = "/usr/src/app/data/example/t_shirt.png"
    cloth_im = Image.open(cloth_path)
    cloth_bytes = bc.image_to_bytes(cloth_im)
    out_cloth_result = cp.consistent_forward(cloth_bytes)
    out_cloth_bytes = out_cloth_result["cloth"]
    out_cloth = bc.bytes_to_image(out_cloth_bytes)
    out_cloth.save("/usr/src/app/volume/data/no_background/cloth_prepr_ex.png")


    hp = HumanProcessor()
    human_path = "/usr/src/app/data/example/human-dc.jpg"
    human_im = Image.open(human_path)
    human_bytes = bc.image_to_bytes(human_im)
    result_human = hp.consistent_forward(human_bytes)


    out_pose = bc.bytes_to_image(result_human["pose"])
    out_pose.save("/usr/src/app/volume/data/pose/pose_aggr.png")

    out_keypoints_bytes = result_human["keypoints_json"]
    out_keypoints = bc.bytes_to_json(out_keypoints_bytes)
    import json
    with open("/usr/src/app/volume/data/pose/keypoints1.json", "w") as f: 
        json.dump(out_keypoints, f)
    # print(out_keypoints)
    parsed_human = bc.bytes_to_image(result_human["parse_orig"])
    parsed_human.save("/usr/src/app/volume/data/parsed/parsed_human_aggregator.png")


