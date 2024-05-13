import os
from enum import Enum

import cv2 
import numpy as np
from skimage import io
import torch
from PIL import Image
from torch import nn
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from app.pkg.ml.try_on.preprocessing.RMBG.briarmbg import BriaRMBG
from app.pkg.ml.try_on.preprocessing.RMBG.utilities import preprocess_image, postprocess_image
from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_pipeline import SegformerSAM_Pipeline
from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_strategies import PointsFormingSamStrategies

from app.pkg.settings import settings

torch.hub.set_dir(settings.ML.WEIGHTS_PATH)
os.environ['TRANSFORMERS_CACHE'] = str(settings.ML.WEIGHTS_PATH)
os.environ['HF_HOME'] = str(settings.ML.WEIGHTS_PATH)


class BackgroundModels(Enum):
    BriaRMBG = "BriaRMBG"
    SegFormerB3 = "sayeed99/segformer_b3_clothes"
    SamPipeline = "SamPipeline"


class ClothPreprocessor:
    def __init__(self,
                 model_type:BackgroundModels = BackgroundModels.BriaRMBG,
                 lightweight=False,
                 ):
        """
        model_type - type of segm model
        lightweight - is need to use light SAM model 
        """
        self.model_type = model_type
 
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if model_type == BackgroundModels.SegFormerB3:
            self.net = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
            self.processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
            self.net = self.net.to(self.device)

        elif model_type == BackgroundModels.BriaRMBG:
            self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
            self.net = self.net.to(self.device)

        elif model_type == BackgroundModels.SamPipeline:
            self.net = SegformerSAM_Pipeline(lightweight)

        else:
            raise ValueError("Not valid cut model name")


        # prepare input
        self.model_input_size = [1024,1024]
    
    def remove_background(self, pil_image:Image,
                          save_mask=False,
                          point_sam_strategy: PointsFormingSamStrategies \
                            = PointsFormingSamStrategies.strategy_0,
                          )->dict:
        """
        image - pil.image with cloth (to load)
        save_mask - is need to save cloth mask.
        point_sam_strategy - strategy to form points (uses only in SAM pipeline)

        Returns {"cloth_no_background":no_bg_image,
                 "cloth_mask": pil_im}
        """
        image = np.array(pil_image.convert('RGB'))
        # convert('RGB') is for images with h,w,4 shape

        orig_im_size = image.shape[0:2]


        with torch.no_grad():
            # inference

            if self.model_type == BackgroundModels.BriaRMBG:
                prep_image = preprocess_image(image, self.model_input_size)

                result = self.net(prep_image.to(self.device))
                # post process
                cloth_mask = postprocess_image(result[0][0], orig_im_size)
                del result

            elif self.model_type == BackgroundModels.SegFormerB3:
                inputs = self.processor(image, return_tensors="pt").to("cuda")
                outputs = self.net(**inputs)
                logits = outputs.logits.cpu().detach()

                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=pil_image.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )

                pred_seg = upsampled_logits.argmax(dim=1)[0]

                cloth_mask = (pred_seg!=0).numpy()

                # clear memory
                del outputs
            elif self.model_type == BackgroundModels.SamPipeline:
                cloth_mask_t = self.net.forward(
                    image,
                    save_meta=False,
                    point_sam_strategy=point_sam_strategy,
                )
                cloth_mask = cloth_mask_t.numpy() 

        pil_im = Image.fromarray(cloth_mask)

        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0 ,0 ,0))

        orig_image = Image.fromarray(image[:,:,:])

        no_bg_image.paste(orig_image, mask=pil_im)
        result = {"cloth_no_background":no_bg_image}

        if save_mask:
            result["cloth_mask"] = pil_im
        return result

    def replace_background_from_mask(self, image, mask,
                           color=(255,255,255)):
        """
        image - pil image with cloth
        mask - pil image with mask cloth
        """
        image = image.convert('RGB')
        
        no_bg_image = Image.new("RGB", mask.size, color)
        no_bg_image.paste(image, mask=mask)
        #no_bg_image.save(save_path)
        return no_bg_image

    @staticmethod
    def replace_background_RGBA(rgba_image,
                           color=(255,255,255)):
        """
        rgba_image - pil image with cloth in rgba format
        color - color to replace background
        """
        # Create a new RGB image with the desired background color
        rgb_image = Image.new("RGB", rgba_image.size, color)

        # Paste the RGBA image onto the RGB image using the mask
        rgb_image.paste(rgba_image, mask=rgba_image)
        return rgb_image

    def crop_and_pad(self, image, pad=10):
        """
        image - pil image with cloth
        """ 
        image = np.array(image)#io.imread(input_path)
        image_last_dim = image.shape[2]
        mask = image.sum(axis=2)>100

        nonzero_coords = np.nonzero(mask)
        min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
        min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])

        # Crop the image based on the bounding box coordinates
        cropped_image = image[min_row:max_row+1, min_col:max_col+1]
        crop_shape = cropped_image.shape
        # Padding
        pad_line = np.zeros((pad, crop_shape[1], image_last_dim), dtype=np.uint8) # 3
        padded_image = np.concatenate((pad_line, cropped_image, pad_line), axis=0)
        pad_column = np.zeros((padded_image.shape[0],pad, image_last_dim), dtype=np.uint8) # 3
        padded_image = np.concatenate((pad_column, padded_image, pad_column), axis=1)

        pil_im = Image.fromarray(padded_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))

        orig_image = Image.fromarray(padded_image)
        no_bg_image.paste(orig_image.convert("RGBA"), mask=pil_im.convert("RGBA"))
        return no_bg_image

    def __call__(self,
                 cloth_im,
                 point_sam_strategy: PointsFormingSamStrategies \
                    = PointsFormingSamStrategies.strategy_0,

                 ):
        """
        cloth_im - pil image of cloth
        """
        im_no_background = self.remove_background(cloth_im,
                                                  save_mask=False,
                                                  point_sam_strategy=point_sam_strategy)["cloth_no_background"]
        crop_and_pad = self.crop_and_pad(im_no_background)
        return crop_and_pad
