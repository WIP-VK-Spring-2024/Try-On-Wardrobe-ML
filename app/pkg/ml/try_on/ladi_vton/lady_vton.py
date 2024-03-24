import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torchvision import transforms


from PIL import Image

import numpy as np
#from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from app.pkg.ml.try_on.ladi_vton.src.dataset.dresscode import DressCodeDataset
from app.pkg.ml.try_on.ladi_vton.src.dataset.vitonhd import VitonHDDataset
from app.pkg.ml.try_on.ladi_vton.src.models.AutoencoderKL import AutoencoderKL
from app.pkg.ml.try_on.ladi_vton.src.utils.encode_text_word_embedding import encode_text_word_embedding
from app.pkg.ml.try_on.ladi_vton.src.utils.set_seeds import set_seed
from app.pkg.ml.try_on.ladi_vton.src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline
from app.pkg.ml.try_on.ladi_vton.lady_vton_prepr import LadyVtonInputPreprocessor

from app.pkg.settings import settings

#PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
torch.hub.set_dir(settings.ML.WEIGHTS_PATH)
os.environ['TRANSFORMERS_CACHE'] = str(settings.ML.WEIGHTS_PATH)# '/usr/src/app/app/pkg/ml/weights'
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")



class LadyVton(torch.nn.Module):
    """
    Virtual try on model implementation    
    """
    def __init__(self, num_inference_steps=20):
        super(LadyVton, self).__init__()
        self.weight_dtype = torch.float32
        self.data_prepr = LadyVtonInputPreprocessor()
        self.device = "cuda:0"
        self.dataset = "dresscode"
        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        self.enable_xformers_memory_efficient_attention = True
        self.seed = 42
        self.num_vstar = 16 # Number of predicted v* images to use
        self.guidance_scale = 7.5
        self.num_inference_steps = num_inference_steps
        self.setup_models()

    def setup_models(self):
        self.val_scheduler = DDIMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        self.val_scheduler.set_timesteps(50, device=self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae")
        self.vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")

        # Load the trained models from the hub
        self.unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet',
                            dataset=self.dataset)
        self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=self.dataset)
        self.inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter',
                                        dataset=self.dataset)
        self.tps, self.refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module',
                                        dataset=self.dataset)

        self.int_layers = [1, 2, 3, 4, 5]

        # Enable xformers memory efficient attention if requested
        if self.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.emasc.to(self.device, dtype=self.weight_dtype)
        self.inversion_adapter.to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.tps.to(self.device, dtype=torch.float32)
        self.refinement.to(self.device, dtype=torch.float32)
        self.vision_encoder.to(self.device, dtype=self.weight_dtype)

        self.text_encoder.eval()
        self.vae.eval()
        self.emasc.eval()
        self.inversion_adapter.eval()
        self.unet.eval()
        self.tps.eval()
        self.refinement.eval()
        self.vision_encoder.eval()

        self.val_pipe = StableDiffusionTryOnePipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.val_scheduler,
            emasc=self.emasc,
            emasc_int_layers=self.int_layers,
        ).to(self.device)

        # if error in generator initialization occures, replace self.device to "cuda"
        self.generator = torch.Generator(self.device).manual_seed(self.seed)


    def forward(self, input_data):
       # input_data = self.data_prepr.preprocess_input(input_data)
    
        model_img = input_data["image"].to(device=self.device, dtype=self.weight_dtype).unsqueeze(0)
        mask_img = input_data["inpaint_mask"].to(device=self.device, dtype=self.weight_dtype).unsqueeze(0)

        pose_map = input_data["pose_map"].to(device=self.device, dtype=self.weight_dtype).unsqueeze(0)
        category = input_data["category"]
        cloth = input_data["cloth"].to(device=self.device, dtype=self.weight_dtype).unsqueeze(0)
        im_mask = input_data['im_mask'].to(device=self.device, dtype=self.weight_dtype).unsqueeze(0)

        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)
        # print(low_im_mask.shape, low_pose_map.shape, )
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = self.tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(512, 384),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth.to(torch.float32), highres_grid.to(torch.float32), padding_mode='border')

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = self.refinement(warped_cloth.to(torch.float32))
        warped_cloth = warped_cloth.clamp(-1, 1)
        warped_cloth = warped_cloth.to(self.weight_dtype)

        # Get the visual features of the in-shop cloths
        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                               antialias=True).clamp(0, 1)
        processed_images = self.processor(images=input_image, return_tensors="pt")
        clip_cloth_features = self.vision_encoder(
            processed_images.pixel_values.to(model_img.device, dtype=self.weight_dtype)).last_hidden_state

        # Compute the predicted PTEs
        word_embeddings = self.inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.num_vstar, -1))

        category_text = {
            'dresses': 'a dress',
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',

        }
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * self.num_vstar}' for
                category in [input_data['category']]]#[batch['category']]]

        # Tokenize text
        tokenized_text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(word_embeddings.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(self.text_encoder, tokenized_text,
                                                           word_embeddings, self.num_vstar).last_hidden_state

        # Generate images
        generated_images = self.val_pipe(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=self.guidance_scale,
            num_images_per_prompt=1,
            generator=self.generator,
            cloth_input_type='warped',
            num_inference_steps=self.num_inference_steps
        ).images

        # Save images
        # for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
        #     if not os.path.exists(os.path.join(save_dir, cat)):
        #         os.makedirs(os.path.join(save_dir, cat))

        #     if args.use_png:
        #         name = name.replace(".jpg", ".png")
        #         gen_image.save(
        #             os.path.join(save_dir, cat, name))
        #     else:
        #         gen_image.save(
        #             os.path.join(save_dir, cat, name), quality=95)
        #generated_images[0].save(save_path)
        return generated_images[0]

if __name__ == '__main__':

    input_data = {
        "category": "upper_body",

    }

    # path to (specifically) resized person image
    human_path = "/usr/src/app/volume/data/resized/human_resized.png"
    input_data["image_human_orig"] = Image.open(human_path).convert('RGB')

    # path to parsed human. Converts into inpaint_mask
    parsed_human_path = "/usr/src/app/volume/data/parsed/parsed_human.png"
    input_data["parse_orig"] = Image.open(parsed_human_path)

    pose_human_im_path = "/usr/src/app/volume/data/pose/posed_human.png"
    
    
    key_points_path = "/usr/src/app/volume/data/pose/keypoints.json"
    # pose_label = input_data["keypoints_json"]
    with open(key_points_path, 'r') as f:
        pose_label = json.load(f)
    input_data['keypoints_json'] = pose_label
   
    # cloth without background
    cloth_path = "/usr/src/app/volume/data/no_background/shirt_white_back.png"
    input_data["cloth"] = Image.open(cloth_path)


    lv_prep = LadyVtonInputPreprocessor()
    lv_prep(input_data)
   
    lv = LadyVton()
