import os

import cv2 
import numpy as np
from skimage import io
import torch
from PIL import Image

from app.pkg.ml.try_on.preprocessing.RMBG.briarmbg import BriaRMBG
from app.pkg.ml.try_on.preprocessing.RMBG.utilities import preprocess_image, postprocess_image

from app.pkg.settings import settings

torch.hub.set_dir(settings.ML.WEIGHTS_PATH)
os.environ['TRANSFORMERS_CACHE'] = str(settings.ML.WEIGHTS_PATH)
os.environ['HF_HOME'] = str(settings.ML.WEIGHTS_PATH)

class ClothPreprocessor:
    def __init__(self):
        self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # prepare input
        self.model_input_size = [1024,1024]
    
    def remove_background(self, image:Image,
                          save_mask=False,
                          )->dict:
        """
        image - pil.image with cloth (to load)
        save_mask - is need to save cloth mask.

        Returns {"cloth_no_background":no_bg_image,
                 "cloth_mask": pil_im}
        """
        image = np.array(image.convert('RGB')) 
        # convert('RGB') is for images with h,w,4 shape
        
        orig_im_size = image.shape[0:2]

        prep_image = preprocess_image(image, self.model_input_size)

        with torch.no_grad(): 
            # inference
            result = self.net(prep_image.to(self.device))
            # post process
            result_image = postprocess_image(result[0][0], orig_im_size)
            # clear memory
            del result        

        pil_im = Image.fromarray(result_image[:,:])

        # if background_color:
        #     no_bg_image = Image.new("RGB", pil_im.size, background_color)
            
        
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0 ,0 ,0))

        orig_image = Image.fromarray(image[:,:,:])

        no_bg_image.paste(orig_image, mask=pil_im)
        result = {"cloth_no_background":no_bg_image}
        #no_bg_image.save(output_path)
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

    def __call__(self, cloth_im):
        """
        cloth_im - pil image of cloth
        """
        im_no_background = self.remove_background(cloth_im, save_mask=False)["cloth_no_background"]
        crop_and_pad = self.crop_and_pad(im_no_background)
        return crop_and_pad


if __name__ == '__main__':
    cp = ClothPreprocessor()
    orig_image = Image.open('/usr/src/app/data/example/t_shirt.png')

    res = cp(orig_image)
    res.save("/usr/src/app/volume/data/no_background/cloth_prepr_ex.png")
    # im_no_background = cp.remove_background(orig_image)

    # cp.crop_and_pad(im_no_background, "/usr/src/app/volume/data/no_background/t_shirt_rc.png")