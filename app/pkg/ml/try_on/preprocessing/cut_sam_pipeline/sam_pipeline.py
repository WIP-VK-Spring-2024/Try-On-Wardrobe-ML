import numpy as np
import torch
from torch import nn
from PIL import Image
from transformers import SamModel, SamProcessor
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_strategies import PointsFormingSamStrategies
from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_former import KeyPointsFormer

class SegformerSAM_Pipeline:
    def __init__(self, lightweight=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pre_segm_model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes").to(self.device)
        self.pre_segm_processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")

        if lightweight:
            model_name = "Zigeng/SlimSAM-uniform-50"#"facebook/sam-vit-huge"
        else:
            model_name = "facebook/sam-vit-huge"

        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)

        self.keypoints_former = KeyPointsFormer()
        # prepare input
        self.model_input_size = [1024,1024]

    def forward(self,
                pil_image:Image,
                point_sam_strategy: PointsFormingSamStrategies \
                    = PointsFormingSamStrategies.strategy_0,
                save_meta = False,
                )->dict:
        """
        image - pil.image with cloth (to load)
        save_mask - is need to save cloth mask.

        Returns {"cloth_no_background":no_bg_image,
                 "cloth_mask": pil_im}
        """
        image = np.array(pil_image.convert('RGB'))

        with torch.no_grad():
            inputs = self.pre_segm_processor(image, return_tensors="pt").to(self.device)
            outputs = self.pre_segm_model(**inputs)
            logits = outputs.logits.cpu().detach()

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=pil_image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            pred_seg = upsampled_logits.argmax(dim=1)[0]

            segformer_cloth_mask = (pred_seg!=0).numpy() 

            # clear memory
            del outputs

        sam_keypoints = self.keypoints_former(segformer_cloth_mask, point_sam_strategy)

        inputs = self.processor(image, input_points=sam_keypoints, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, )

            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            scores = outputs.iou_scores

            mask_index = scores.argmax()

            pred_seg = masks[0][0, mask_index, :, :]

            cloth_mask = pred_seg!=0

            del inputs
            del outputs
        return cloth_mask

        # pil_im = Image.fromarray(cloth_mask.numpy())           

        # no_bg_image = Image.new("RGBA", pil_im.size, (0, 0 ,0 ,0))

        # orig_image = Image.fromarray(image)

        # no_bg_image.paste(orig_image, mask=pil_im)
        # result = {"cloth_no_background":no_bg_image}
        # if save_meta:
        #     result["sam_cloth_mask"] = pil_im
        #     result['segformer_cloth_mask'] = Image.fromarray(segformer_cloth_mask)
        #     result['sam_keypoints'] = sam_keypoints

        # return result
