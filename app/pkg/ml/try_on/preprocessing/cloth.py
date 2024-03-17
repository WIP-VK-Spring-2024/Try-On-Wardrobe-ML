import cv2 
import numpy as np
from skimage import io
import torch, os
from PIL import Image
from RMBG.briarmbg import BriaRMBG
from RMBG.utilities import preprocess_image, postprocess_image


class ClothPreprocessor:
    def __init__(self):
        self.net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        # prepare input
        self.model_input_size = [1024,1024]
    
    def remove_background(self, input_path, output_path):
        """
        input_path - path to resized image (to load)
        keypoint_output_path - path to json (to save)
        output_path - path to img (to save)
        """
        image = io.imread(input_path)
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
        # tops = result_image.argmax(0)#.tolist() #,#result_image.argmin(0) )
        # top_idx = tops[tops > 0].min()

#        print(tops)
#        print(tops.min())
        # save result
        
        pil_im = Image.fromarray(result_image[:,:])
#        print(result_image.shape, pil_im.shape)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        orig_image = Image.fromarray(image[:,:,:])

        no_bg_image.paste(orig_image, mask=pil_im)
        no_bg_image.save(output_path)

    def crop_and_pad(self, input_path, output_path, pad=10):
        image = io.imread(input_path)
        mask = image.sum(axis=2)>100 

        nonzero_coords = np.nonzero(mask)
        min_row, max_row = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
        min_col, max_col = np.min(nonzero_coords[1]), np.max(nonzero_coords[1])

        # Crop the image based on the bounding box coordinates
        cropped_image = image[min_row:max_row+1, min_col:max_col+1]
        crop_shape = cropped_image.shape
        # Padding
        pad_line = np.zeros((pad, crop_shape[1], 4)) # 3
        print(pad_line.shape, crop_shape)
        padded_image = np.concatenate((pad_line, cropped_image, pad_line), axis=0)
        print(padded_image.dtype, cropped_image.dtype)
        # bottom_idx = bottoms[bottoms > 0].min()
    
        pil_im = Image.fromarray(padded_image)
        #       print(result_image.shape, pil_im.shape)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        orig_image = Image.fromarray(padded_image)

        no_bg_image.paste(orig_image, mask=pil_im)
        no_bg_image.save(output_path)





if __name__ == '__main__':
    cp = ClothPreprocessor()
    # cp.remove_background('/usr/src/app/data/example/t_shirt.png',
    #    '/usr/src/app/volume/data/no_background/t_shirt.png'
    #    )
    cp.crop_and_pad('/usr/src/app/volume/data/no_background/t_shirt.png', "/usr/src/app/volume/data/no_background/t_shirt_rc.png")