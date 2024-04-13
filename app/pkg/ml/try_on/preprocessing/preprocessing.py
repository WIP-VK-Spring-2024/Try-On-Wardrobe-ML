import cv2
from PIL import Image
import numpy as np


class Resizer:
    """
    Custom resizer. It stays original image w/h proportion.
    And resize it to desired shape
    """    
    def __init__(self):
        self.RESIZE_WIDTH = 384
        self.RESIZE_HEIGHT = 512


    def __call__(self, image:Image, color=(0,0,0)):
        """
        image - pil image for resize
        """
 
        image = np.array(image)
        resized_image = resize_with_pad(image, self.RESIZE_WIDTH, self.RESIZE_HEIGHT, color=color)

        assert resized_image.shape[:2] == (self.RESIZE_HEIGHT, self.RESIZE_WIDTH )
        
        return Image.fromarray(resized_image)
        #cv2.imwrite(output_path, resized_image)


def resize_with_pad(im, target_width, target_height, color=(0,0,0)):
    '''
    Resize image keeping ratio and using white background.
    '''
    ori_shape = im.shape
    target_ratio = target_height / target_width
    im_ratio = ori_shape[0] / ori_shape[1]
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = cv2.resize(im, (resize_width, resize_height), interpolation = cv2.INTER_AREA)

    top_pad = int(max(0,(target_height-resize_height)+0.001)/2) # its pad from top
    bottom_pad = round(max(0,(target_height-resize_height)+0.001)/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

    left_pad = int(max(0,(target_width-resize_width)+0.001)/2) # its pad from left side
    right_pad = round(max(0,(target_width-resize_width)+0.001)/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

    padded_image = cv2.copyMakeBorder(image_resize, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=color)
 #   padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    return padded_image


if __name__ == '__main__':
    prepr = Resizer()
    prepr('/usr/src/app/data/example/human.png', '/usr/src/app/volume/data/resized/resized_human.png')
