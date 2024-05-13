import io
from typing import Dict, Union

from PIL import Image

from app.pkg.ml.buffer_converters import BytesConverter

from app.pkg.ml.try_on.preprocessing.preprocessing import Resizer

from app.pkg.ml.try_on.preprocessing.pose import PoseEstimation
from app.pkg.ml.try_on.preprocessing.cloth import ClothPreprocessor, BackgroundModels
from app.pkg.ml.try_on.preprocessing.human_parsing import HumanParsing
from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_strategies import PointsFormingSamStrategies


class BaseProcessor:
    def __init__(self):
        self.bytes_converter = BytesConverter()


class ClothProcessor(BaseProcessor):
    """
    Class for processing clothes. Puts into worker
    """
    def __init__(self,
                 model_type=BackgroundModels.SamPipeline,
                 light_weight=False):
        """
        model_type - type of segmentaton model used
        light_weight - is need to use light weight version of model (only for SAM pipeline)
        """
        super().__init__()
        self.model_type = model_type
        self.model_background = ClothPreprocessor(model_type, light_weight)
        self.usage_counter = 0
        self.strategies = [strategy.value for strategy in PointsFormingSamStrategies]

    def consistent_forward(self,
                           image_bytes:io.BytesIO,
                           point_sam_strategy=None,

                           ) -> Dict[str, io.BytesIO]:
        """
        Processes cloth image
        Removes background from input image buffer

        Args:
            image_bytes - bytes of cloth image
        
        Returns:
            result - dict with Dict[str, io.BytesIO] format
        """
        image = self.bytes_converter.bytes_to_image(image_bytes)
        
        # selecting strategy to get points
        if point_sam_strategy is None:
            if self.usage_counter >= len(self.strategies):
                self.usage_counter = 0

            point_sam_strategy = self.strategies[self.usage_counter]
        else:
            point_sam_strategy = point_sam_strategy.value
        self.usage_counter += 1


        no_background_image = self.model_background(
            cloth_im=image,
            point_sam_strategy=point_sam_strategy)
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

        result = self.process(image)

        return result                

    def process(self, image:Image):
        human_resized = self.resizer(image, color=(255,255,255))
        result = {}
        pose_out, keypoints_json_dict = self.model_pose_estim(human_resized)
        result["pose"] = self.bytes_converter.image_to_bytes(pose_out)
        result["keypoints_json"] = self.bytes_converter.json_to_bytes(keypoints_json_dict)

        parsed_human = self.model_human_parsing(human_resized)
        result["parse_orig"] = self.bytes_converter.image_to_bytes(parsed_human)
        result["image_human_orig"] = self.bytes_converter.image_to_bytes(human_resized)
        return result

