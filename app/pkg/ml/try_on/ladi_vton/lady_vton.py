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



class LadyVton:
    """
    Virtual try on model implementation    
    """
    def __init__(self, num_inference_steps=20):
        self.weight_dtype = torch.float32
        self.data_prepr = LadyVtonInputPreprocessor()
        self.device = "cuda:1"
        self.dataset = "dresscode"
        self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        self.enable_xformers_memory_efficient_attention = True
        self.seed = 42
        self.num_vstar = 16 # Number of predicted v* images to use
        self.guidance_scale = 7.5
        self.num_inference_steps = num_inference_steps
        # self.setup_models()

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
        self.emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=args.dataset)
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


    def forward(self, input_data, save_path):
        input_data = self.data_prepr.preprocess_input(input_data)
    
        model_img = input_data["image"].to(self.weight_dtype)
        mask_img = input_data["inpaint_mask"].to(self.weight_dtype)
        if mask_img is not None:
            mask_img = mask_img.to(self.weight_dtype)
        pose_map = input_data["pose_map"].to(self.weight_dtype)
        category = input_data["category"]
        cloth = input_data["cloth"].to(self.weight_dtype)
        im_mask = input_data['im_mask'].to(self.weight_dtype)

        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = self.tps(low_cloth.to(torch.float32), agnostic.to(torch.float32))

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

# def parse_args():
#     parser = argparse.ArgumentParser(description="Full inference script")

#     parser.add_argument(
#         "--pretrained_model_name_or_path",
#         type=str,
#         default="stabilityai/stable-diffusion-2-inpainting",
#         help="Path to pretrained model or model identifier from huggingface.co/models.",
#     )

#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="Path to the output directory",
#     )

#     parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
#     parser.add_argument("--batch_size", type=int, default=8, help="Batch size to use.")

#     parser.add_argument(
#         "--mixed_precision",
#         type=str,
#         default=None,
#         choices=["no", "fp16", "bf16"],
#         help=(
#             "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
#             " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
#             " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
#         ),
#     )

#     parser.add_argument(
#         "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
#     )

#     parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
#     parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

#     parser.add_argument("--num_workers", type=int, default=8,
#                         help="The name of the repository to keep in sync with the local `output_dir`.")

#     parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
#     parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
#     parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
#     parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
#     parser.add_argument("--use_png", default=False, action="store_true")
#     parser.add_argument("--num_inference_steps", default=50, type=int)
#     parser.add_argument("--guidance_scale", default=10, type=float)
#     parser.add_argument("--compute_metrics", default=False, action="store_true")

#     args = parser.parse_args()
#     env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     if env_local_rank != -1 and env_local_rank != args.local_rank:
#         args.local_rank = env_local_rank

#     return args


# @torch.inference_mode()
# def main():
  
#     device = 'cpu'#accelerator.device
#     # If passed along, set the training seed now.
#         # Cast to weight_dtype
#     weight_dtype = torch.float32
#     # if args.mixed_precision == 'fp16':
#     #     weight_dtype = torch.float16
        
#     # Load scheduler, tokenizer and models.
#     default_path = "stabilityai/stable-diffusion-2-inpainting"
#     # Load scheduler, tokenizer and models.
#     # val_scheduler = DDIMScheduler.from_pretrained(default_path, subfolder="scheduler")
#     # val_scheduler.set_timesteps(50, device=device)

#     # text_encoder = CLIPTextModel.from_pretrained(default_path, subfolder="text_encoder")
#     # vae = AutoencoderKL.from_pretrained(default_path, subfolder="vae")
#     # vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     # processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
#     # tokenizer = CLIPTokenizer.from_pretrained(default_path, subfolder="tokenizer")

#     # text_encoder.to(device, dtype=weight_dtype)
#     # vae.to(device, dtype=weight_dtype)
#     # vision_encoder.to(device, dtype=weight_dtype)

#     # dataset = 'dresscode'
#     # # Load the trained models from the hub
#     # unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet', dataset=dataset)
#     # emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=dataset)
#     # inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter', dataset=dataset)
#     # tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', dataset=dataset)
    
#     # unet.to(device, dtype=weight_dtype)
#     # emasc.to(device, dtype=weight_dtype)
#     # inversion_adapter.to(device, dtype=weight_dtype)
#     # tps.to(device, dtype=weight_dtype)
#     # refinement.to(device, dtype=weight_dtype)

#     # int_layers = [1, 2, 3, 4, 5]


#     # # Set to eval mode
#     # text_encoder.eval()
#     # vae.eval()
#     # emasc.eval()
#     # inversion_adapter.eval()
#     # unet.eval()
#     # tps.eval()
#     # refinement.eval()
#     # vision_encoder.eval()

#     # # Create the pipeline
#     # val_pipe = StableDiffusionTryOnePipeline(
#     #     text_encoder=text_encoder,
#     #     vae=vae,
#     #     tokenizer=tokenizer,
#     #     unet=unet,
#     #     scheduler=val_scheduler,
#     #     emasc=emasc,
#     #     emasc_int_layers=int_layers,
#     # ).to(device)

#     # Prepare the dataloader and create the output directory
#     #test_dataloader = accelerator.prepare(test_dataloader)
#     #save_dir = os.path.join(args.output_dir, args.test_order)
#     #os.makedirs(save_dir, exist_ok=True)
#     # generator = torch.Generator(device).manual_seed()

#     # # Generate the images
#     for idx, batch in enumerate(tqdm(test_dataloader)):
#         model_img = batch.get("image").to(weight_dtype)
#         mask_img = batch.get("inpaint_mask").to(weight_dtype)
#         if mask_img is not None:
#             mask_img = mask_img.to(weight_dtype)
#         pose_map = batch.get("pose_map").to(weight_dtype)
#         category = batch.get("category")
#         cloth = batch.get("cloth").to(weight_dtype)
#         im_mask = batch.get('im_mask').to(weight_dtype)

#     #     # Generate the warped cloth
#     #     # For sake of performance, the TPS parameters are predicted on a low resolution image

#     #     low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
#     #                                                          torchvision.transforms.InterpolationMode.BILINEAR,
#     #                                                          antialias=True)
#     #     low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
#     #                                                            torchvision.transforms.InterpolationMode.BILINEAR,
#     #                                                            antialias=True)
#     #     low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
#     #                                                             torchvision.transforms.InterpolationMode.BILINEAR,
#     #                                                             antialias=True)
#     #     agnostic = torch.cat([low_im_mask, low_pose_map], 1)
#     #     low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

#     #     # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
#     #     highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
#     #                                                             size=(512, 384),
#     #                                                             interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
#     #                                                             antialias=True).permute(0, 2, 3, 1)

#     #     warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

#     #     # Refine the warped cloth using the refinement network
#     #     warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
#     #     warped_cloth = refinement(warped_cloth)
#     #     warped_cloth = warped_cloth.clamp(-1, 1)

#     #     # Get the visual features of the in-shop cloths
#     #     input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
#     #                                                            antialias=True).clamp(0, 1)
#     #     processed_images = processor(images=input_image, return_tensors="pt")
#     #     clip_cloth_features = vision_encoder(
#     #         processed_images.pixel_values.to(model_img.device, dtype=weight_dtype)).last_hidden_state

#     #     # Compute the predicted PTEs
#     #     word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
#     #     word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], args.num_vstar, -1))

#     #     category_text = {
#     #         'dresses': 'a dress',
#     #         'upper_body': 'an upper body garment',
#     #         'lower_body': 'a lower body garment',

#     #     }
#     #     text = [f'a realistic photo of a model wearing {category_text[category]} {" $ " * args.num_vstar}' for
#     #             category in batch['category']]

#     #     # Tokenize text
#     #     tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
#     #                                truncation=True, return_tensors="pt").input_ids
#     #     tokenized_text = tokenized_text.to(word_embeddings.device)

#     #     # Encode the text using the PTEs extracted from the in-shop cloths
#     #     encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
#     #                                                        word_embeddings, args.num_vstar).last_hidden_state

#     #     # negative_prompt = ['fake unrealistic image,stock footage watermark, close up, grainy, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, out of frame, ugly, extra limbs, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, cross-eyed, body out of frame, mutated, bad body, closed eyes']

#     #     # Generate images
#     #     generated_images = val_pipe(
#     #         image=model_img,
#     #         mask_image=mask_img,
#     #         pose_map=pose_map,
#     #         warped_cloth=warped_cloth,
#     #         prompt_embeds=encoder_hidden_states,
#     #         height=512,
#     #         width=384,
#     #         guidance_scale=args.guidance_scale,
#     #         num_images_per_prompt=1,
#     #         generator=generator,
#     #         cloth_input_type='warped',
#     #         num_inference_steps=args.num_inference_steps
#     #     ).images

#     #     # Save images
#     #     for gen_image, cat, name in zip(generated_images, category, batch["im_name"]):
#     #         if not os.path.exists(os.path.join(save_dir, cat)):
#     #             os.makedirs(os.path.join(save_dir, cat))

#     #         if args.use_png:
#     #             name = name.replace(".jpg", ".png")
#     #             gen_image.save(
#     #                 os.path.join(save_dir, cat, name))
#     #         else:
#     #             gen_image.save(
#     #                 os.path.join(save_dir, cat, name), quality=95)

#     # # Free up memory
#     # del val_pipe
#     # del text_encoder
#     # del vae
#     # del emasc
#     # del unet
#     # del tps
#     # del refinement
#     # del vision_encoder
#     # torch.cuda.empty_cache()

# if __name__ == "__main__":
#     main()