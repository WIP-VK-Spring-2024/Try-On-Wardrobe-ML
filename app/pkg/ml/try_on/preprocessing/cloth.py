import cv2 
import numpy as np
from skimage import io
import torch
from PIL import Image

from app.pkg.ml.try_on.preprocessing.RMBG.briarmbg import BriaRMBG
from app.pkg.ml.try_on.preprocessing.RMBG.utilities import preprocess_image, postprocess_image


class ClothPreprocessor:
    def __init__(self):
        self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # prepare input
        self.model_input_size = [1024,1024]
    
    def remove_background(self, input_path,
                          output_path,
                          background_path=None,
                          background_color=None):
        """
        input_path - path to resized image (to load)
        keypoint_output_path - path to json (to save)
        output_path - path to img (to save)
        background_path - path to save cloth mask
        """
        image = np.array(Image.open(input_path).convert('RGB')) # io.imread 
        # convert('RGB') is for images with h,w,4 shape
        
        orig_im_size = image.shape[0:2]

        if image is None:
            raise Exception(f"Image {input_path} is not found for pose estimation")
        
        prep_image = preprocess_image(image, self.model_input_size)

        with torch.no_grad(): 
            # inference
            result = self.net(prep_image.to(self.device))
            # post process
            result_image = postprocess_image(result[0][0], orig_im_size)
            # clear memory
            del result        
        # selecting images to crop only cloth
        
        pil_im = Image.fromarray(result_image[:,:])
#        print(result_image.shape, pil_im.shape)
        if background_color:
            no_bg_image = Image.new("RGB", pil_im.size, background_color)
            
        else:
            no_bg_image = Image.new("RGBA", pil_im.size, (0, 0 ,0 ,0))

        orig_image = Image.fromarray(image[:,:,:])

        no_bg_image.paste(orig_image, mask=pil_im)
        no_bg_image.save(output_path)
        if background_path:
            pil_im.save(background_path)
        
    def replace_background(self, im_path, mask_path, save_path,
                           color=(255,255,255)):
        image = Image.open(im_path).convert('RGB')
        mask = Image.open(mask_path)

        no_bg_image = Image.new("RGB", mask.size, color)
        no_bg_image.paste(image, mask=mask)
        no_bg_image.save(save_path)


    def crop_and_pad(self, input_path, output_path, pad=10):
        image = np.array(Image.open(input_path))#io.imread(input_path)
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
        #       print(result_image.shape, pil_im.shape)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))

        # if image_last_dim == 4:

        # elif image_last_dim == 3:
        #     no_bg_image = Image.new("RGB", pil_im.size, (0,0,0))
        # else:
        #     raise TypeError("Caught unknown shape image")
        orig_image = Image.fromarray(padded_image)
        # print(np.array(pil_im).shape, np.array(orig_image).shape)
        # print(np.array(no_bg_image).shape)
        no_bg_image.paste(orig_image.convert("RGBA"), mask=pil_im.convert("RGBA"))
        no_bg_image.save(output_path)

if __name__ == '__main__':
    cp = ClothPreprocessor()
    cp.remove_background('/usr/src/app/data/example/t_shirt.png',
       '/usr/src/app/volume/data/no_background/t_shirt.png'
       )
    cp.crop_and_pad('/usr/src/app/volume/data/no_background/t_shirt.png', "/usr/src/app/volume/data/no_background/t_shirt_rc.png")