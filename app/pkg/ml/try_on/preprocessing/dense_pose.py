
import torch
import numpy as np
import os, json, cv2, random

from app.pkg.settings import settings

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import CfgNode, get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from densepose import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.base import CompoundVisualizer
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

class DensePoseEstimation:

    def __init__(self):
        self.WEIGHTS_PATH = f"{settings.ML.WEIGHTS_PATH}/dense_pose.pkl"
        self.cfg_path = "app/pkg/ml/try_on/preprocessing/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        self.setup()
    
    # setup cfg
    def setup(self):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.cfg_path)
        #cfg.merge_from_list(args.opts)
        # if opts:
        #     cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = self.WEIGHTS_PATH
        cfg.freeze()
        self.cfg = cfg
        self.setup_models()

    def setup_models(self):
        self.predictor = DefaultPredictor(self.cfg)
        visualizers = []
        extractors = []
        texture_atlas = None
        texture_atlases_dict = None
        vis = DensePoseResultsFineSegmentationVisualizer(
            cfg=self.cfg,
            texture_atlas=texture_atlas,
            texture_atlases_dict=texture_atlases_dict,
        )
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)

        self.visualizer = CompoundVisualizer(visualizers)
        self.extractor = CompoundExtractor(extractors)

        self.dump_extractor = DensePoseResultExtractor()

    def __call__(self, input_path, output_image_path, output_npz_path):
        """
        input_path - path to resized image (to load)
        output_path - path to img (to save)
        output_npz_path - path to .npz (to save)
        """
        image = cv2.imread(input_path)
        if image is None:
            raise Exception(f"Image {input_path} is not found for pose estimation")
        assert image.shape == (512, 384, 3)
        
        image_data = {"file_name": input_path,
                      "image": image}

        with torch.no_grad():
            outputs = self.predictor(image)["instances"]

            # visual (.png, .jpg, etc.) file creation        
            self.postprocess_outputs(image_data,
                                     outputs,
                                     output_image_path)

            # .npz file creation
            return self.dump_post_process(image_data,
                                   outputs,
                                   output_npz_path)
            
            # checking for correct version of outputs format
            # assert isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput)


    def postprocess_outputs(self, image_data, outputs, out_fname):
        image = cv2.cvtColor(image_data["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = self.extractor(outputs)
        image_vis = self.visualizer.visualize(image, data)
        
        cv2.imwrite(out_fname, image_vis)

    def dump_post_process(self, image_data, outputs, out_fname):
        image_fpath = image_data["file_name"]
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                # if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                #     extractor = DensePoseResultExtractor()
                # elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                #     extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = self.dump_extractor(outputs)[0]
        return result
        # results = [result]
        # with open(out_fname, "wb") as hFile:
        #     torch.save(results, hFile)


# if __name__ == '__main__':
#     dpe = DensePoseEstimation()
#     dpe(
#        "/usr/src/app/volume/data/resized/resized_human.png",
#        "/usr/src/app/volume/data/dense_pose/dense_pose_human.png",
#        "/usr/src/app/volume/data/dense_pose/dense_pose_human.npz",
#        )

# if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
#     print("extractor = DensePoseResultExtractor()")
# elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
#     print("extractor = DensePoseOutputsExtractor()")


