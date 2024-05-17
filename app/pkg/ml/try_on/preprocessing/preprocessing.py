import cv2
from PIL import Image
import numpy as np


class Resizer:
    """
    Custom resizer. It stays original image w/h proportion.
    And resize it to desired shape
    """    
    def __init__(self, try_on_width=384, try_on_height=512):
        self.PREPROC_WIDTH = 384 # preprocessing
        self.PREPROC_HEIGHT = 512

        self.TRY_ON_WIDTH = try_on_width
        self.TRY_ON_HEIGHT = try_on_height
        

    def stretch_resize(self,
                       image:Image,
                       preproc=True):
        if preproc:
            width = self.PREPROC_HEIGHT
            height = self.PREPROC_WIDTH
        else:
            width = self.TRY_ON_HEIGHT
            height = self.TRY_ON_WIDTH

        return image.resize((height, width))

    def remove_borders(self,
                       image: Image,
                       original_image: Image):
        """
        Removes borders from resized image
        image - image with borders (from pad resize)
        original_image - original image (without any borders)
        """
        np_original_image = np.array(original_image)
        np_image = np.array(image)
        np_no_borders = remove_pad(np_image, np_original_image)
        pil_no_borders = Image.fromarray(np_no_borders)
        return pil_no_borders

    def __call__(self, image:Image, color=(0,0,0)):
        """
        image - pil image for resize
        """
 
        image = np.array(image)
        resized_image = resize_with_pad(image,
                                        self.TRY_ON_WIDTH,
                                        self.TRY_ON_HEIGHT,
                                        color=color)

        assert resized_image.shape[:2] == (self.TRY_ON_HEIGHT, self.TRY_ON_WIDTH )
        
        return Image.fromarray(resized_image)


def remove_pad(image:np.ndarray,
               original_image:np.ndarray):
    """
    Removes padding created with resizing
    image - array with padding
    original_image - array without padding (before padding resize)

    """
    image_shape = image.shape
    ori_shape = original_image.shape
    target_height, target_width, _ =  image_shape
    target_ratio = target_height / target_width
    im_ratio = ori_shape[0] / ori_shape[1]
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)

        top_pad = int(max(0,(target_height-resize_height)+0.001)/2) # its pad from top
        bottom_pad = round(max(0,(target_height-resize_height)+0.001)/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

        return image[top_pad:-bottom_pad,:,:]
    elif target_ratio < im_ratio:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

        left_pad = int(max(0,(target_width-resize_width)+0.001)/2) # its pad from left side
        right_pad = round(max(0,(target_width-resize_width)+0.001)/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

        return image[:,left_pad:-right_pad,:]
    else:
        return image

    # image_resize = cv2.resize(im, (resize_width, resize_height), interpolation = cv2.INTER_AREA)



    # padded_image = cv2.copyMakeBorder(image_resize, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=color)
 #   padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    

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
    # prepr = Resizer()
    # prepr('/usr/src/app/data/example/human.png', '/usr/src/app/volume/data/resized/resized_human.png')
    image = Image.open("/usr/src/app/volume/tmp/idm_try_on/3.png")
    orig = Image.open("/usr/src/app/data/human/jennifer_lourence.png")
    no_pad_im = remove_pad(np.array(image), np.array(orig))
    print(no_pad_im.shape)
    Image.fromarray(no_pad_im).save("/usr/src/app/volume/tmp/resize/no_pad.png")