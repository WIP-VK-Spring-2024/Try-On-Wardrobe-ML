import cv2
import os

def resize_with_pad(im, target_width, target_height):
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

    top_pad = int(max(0,(target_height-resize_height))/2) # its pad from top
    bottom_pad = round(max(0,(target_height-resize_height))/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

    left_pad = int(max(0,(target_width-resize_width))/2) # its pad from left side
    right_pad = round(max(0,(target_width-resize_width))/2) # if need pad = 3. It will be int(1.5) + round(1.5) = 3

    padded_image = cv2.copyMakeBorder(image_resize, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255,255,255))
    padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    return padded_image
