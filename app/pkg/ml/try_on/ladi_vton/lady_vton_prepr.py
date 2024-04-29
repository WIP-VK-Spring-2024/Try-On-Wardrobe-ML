import argparse
import json
import os
from pathlib import Path
import copy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torchvision import transforms
# from torchvision.transforms import v2 # 

from PIL import Image, ImageDraw, ImageOps
import cv2

import numpy as np
#from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor
from numpy.linalg import lstsq

from app.pkg.ml.try_on.ladi_vton.src.dataset.dresscode import DressCodeDataset
from app.pkg.ml.try_on.ladi_vton.src.dataset.vitonhd import VitonHDDataset
from app.pkg.ml.try_on.ladi_vton.src.models.AutoencoderKL import AutoencoderKL
from app.pkg.ml.try_on.ladi_vton.src.utils.encode_text_word_embedding import encode_text_word_embedding
from app.pkg.ml.try_on.ladi_vton.src.utils.set_seeds import set_seed
from app.pkg.ml.try_on.ladi_vton.src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline
from app.pkg.ml.try_on.ladi_vton.src.utils.posemap import kpoint_to_heatmap
from app.pkg.models.app.image_category import ImageCategory

# map for human parsing block
label_map={ 
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}
clothes_types = [ImageCategory.DRESSES,
                 ImageCategory.UPPER_BODY,
                 ImageCategory.LOWER_BODY]

class LadyVtonInputPreprocessor:
    def __init__(self,):
        self.width = 384
        self.height = 512
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.radius = 5

    def __call__(self, input_data):
        # TODO: Insert here clever resize
        # because it won't work idealy
        self.prepare_human(input_data)
        self.prepare_cloth(input_data)
        return input_data


    def prepare_human(self, input_data):
        input_data['image'] = self.preprocess_human_orig(input_data['image_human_orig'])
        self.preprocess_human_parsing(input_data)


    def prepare_cloth(self, input_data):
        input_data['cloth'] = self.preprocess_cloth(input_data['cloth'])


    def preprocess_human_orig(self, image):
        # human image preprocessing
        image = image.resize((self.width, self.height),  Image.NEAREST)
        image = self.transform(image)
        return image

    def preprocess_cloth(self, image):
        # human image preprocessing
        image = image.resize((self.width, self.height),  Image.NEAREST)
        image = self.transform(image)
        return image


    def preprocess_human_parsing(self, input_data):
        """
        Releases human parsing and loading keypoints loading        
        """
        image = input_data["image"]
        cloth_type = input_data["category"]

        im_parse = input_data["parse_orig"].resize((self.width, self.height)) 
        parse_array = np.array(im_parse)
        parse_shape = (parse_array > 0).astype(np.float32)

        parse_head = (parse_array == 1).astype(np.float32) + \
                    (parse_array == 2).astype(np.float32) + \
                    (parse_array == 3).astype(np.float32) + \
                    (parse_array == 11).astype(np.float32)

        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["bag"]).astype(np.float32)

        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)
        if  cloth_type == ImageCategory.DRESSES:
            label_cat = 7
            parse_cloth = (parse_array == 7).astype(np.float32)
            parse_mask = (parse_array == 7).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32) + \
                            (parse_array == 13).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif cloth_type == ImageCategory.UPPER_BODY:
            label_cat = 4
            parse_cloth = (parse_array == 4).astype(np.float32)
            parse_mask = (parse_array == 4).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                    (parse_array == label_map["pants"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        elif cloth_type == ImageCategory.LOWER_BODY:
            label_cat = 6
            parse_cloth = (parse_array == 6).astype(np.float32)
            parse_mask = (parse_array == 6).astype(np.float32) + \
                            (parse_array == 12).astype(np.float32) + \
                            (parse_array == 13).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                    (parse_array == 14).astype(np.float32) + \
                                    (parse_array == 15).astype(np.float32)
            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
        else:
            raise NotImplementedError

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()


        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.width // 16, self.height // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
        shape = self.transform2D(parse_shape)  # [-1,1]

        # Load pose points
        pose_label = copy.deepcopy(input_data["keypoints_json"])
        pose_data = pose_label['keypoints'] # be careful on changes here. Pose label shouldn't change
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 4))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        r = self.radius * (self.height / 512.0)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (self.width, self.height))
        neck_draw = ImageDraw.Draw(neck)
        for i in range(point_num):
            one_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
            point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                        'white')
            one_map = self.transform2D(one_map)
            pose_map[i] = one_map[0]

        d = []
        for pose_d in pose_data:
            ux = pose_d[0] / 384.0
            uy = pose_d[1] / 512.0

            # scale posemap points
            px = ux * self.width
            py = uy * self.height

            d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

        pose_map = torch.stack(d)
        input_data['pose_map'] = pose_map

        # just for visualization
        im_pose = self.transform2D(im_pose)

        im_arms = Image.new('L', (self.width, self.height))
        arms_draw = ImageDraw.Draw(im_arms)
        if cloth_type in  clothes_types:
            # belive that prev keypoint json usage didn't change it
            data = copy.deepcopy(input_data["keypoints_json"])
            shoulder_right = np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0)
            shoulder_left = np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0)
            elbow_right = np.multiply(tuple(data['keypoints'][3][:2]), self.height / 512.0)
            elbow_left = np.multiply(tuple(data['keypoints'][6][:2]), self.height / 512.0)
            wrist_right = np.multiply(tuple(data['keypoints'][4][:2]), self.height / 512.0)
            wrist_left = np.multiply(tuple(data['keypoints'][7][:2]), self.height / 512.0)
            if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                        np.uint16).tolist(), 'white', 45, 'curve')
            elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                    arms_draw.line(
                        np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', 45, 'curve')
            else:
                arms_draw.line(np.concatenate(
                    (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                    np.uint16).tolist(), 'white', 45, 'curve')

            hands = np.logical_and(np.logical_not(im_arms), arms)

            if cloth_type in [ImageCategory.DRESSES, ImageCategory.UPPER_BODY]:
                parse_mask += im_arms
                parser_mask_fixed += hands

        # delete neck
        parse_head_2 = torch.clone(parse_head)
        if cloth_type in [ImageCategory.DRESSES, ImageCategory.UPPER_BODY]:
                data = copy.deepcopy(input_data["keypoints_json"])
                points = []
                points.append(np.multiply(tuple(data['keypoints'][2][:2]), self.height / 512.0))
                points.append(np.multiply(tuple(data['keypoints'][5][:2]), self.height / 512.0))
                x_coords, y_coords = zip(*points)
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]
                for i in range(parse_array.shape[1]):
                    y = i * m + c
                    parse_head_2[int(y - 20 * (self.height / 512.0)):, i] = 0

        parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
        parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                np.logical_not(
                                                                    np.array(parse_head_2, dtype=np.uint16))))

        parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)

        parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
        parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
        im_mask = image * parse_mask_total
        input_data['im_mask'] = im_mask
        inpaint_mask = 1 - parse_mask_total

        inpaint_mask = inpaint_mask.unsqueeze(0)
        input_data["inpaint_mask"] = inpaint_mask

        parse_mask_total = parse_mask_total.numpy()
        parse_mask_total = parse_array * parse_mask_total
        parse_mask_total = torch.from_numpy(parse_mask_total)

