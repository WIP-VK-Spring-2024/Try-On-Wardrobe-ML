from typing import List

from PIL import Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoTokenizer
import numpy as np
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation

from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.unet_hacked_tryon import UNet2DConditionModel

from app.pkg.models.app.image_category import ImageCategory
from app.pkg.settings import settings

from app.pkg.logger import get_logger

logger = get_logger(__name__)

base_path = 'yisol/IDM-VTON'



class IDM_VTON(torch.nn.Module):
    """
    Virtual try on model implementation    
    """
    def __init__(self, num_inference_steps=30):
        super(IDM_VTON, self).__init__()
        self.weight_dtype = torch.float16  # torch.float32
        self.device = "cuda:0"
        self.seed = 42
        self.guidance_scale = 2 # 7.5
        self.num_inference_steps = num_inference_steps
        self.WEIGHTS_PATH = settings.ML.WEIGHTS_PATH
        self.DENSE_POSE_WEIGHTS_PATH = f"{self.WEIGHTS_PATH}/dense_pose.pkl"

        self.height= 1024#1024 # (512, 384)
        self.width= 768#768
        self.final_resize = transforms.Resize((self.height, self.width))
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()
        self.setup_models()


    def setup_models(self):

        self.unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device, self.weight_dtype)
        logger.info('Initing tryon. 1/4')

        self.unet.requires_grad_(False)
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        logger.info('Initing tryon. 2/4')

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device, self.weight_dtype)
        logger.info('Initing tryon. 3/4')

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        ).to(self.device, self.weight_dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=self.weight_dtype,
            ).to(self.device, self.weight_dtype)
        self.vae = AutoencoderKL.from_pretrained(base_path,
                                            subfolder="vae",
                                            torch_dtype=self.weight_dtype,
        ).to(self.device, self.weight_dtype)

        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device, self.weight_dtype)
        logger.info('Initing tryon. 4/4')

        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

        self.pipe = TryonPipeline.from_pretrained(
                base_path,
                unet=self.unet,
                vae=self.vae,
                feature_extractor= CLIPImageProcessor(),
                text_encoder = self.text_encoder_one,
                text_encoder_2 = self.text_encoder_two,
                tokenizer = self.tokenizer_one,
                tokenizer_2 = self.tokenizer_two,
                scheduler = self.noise_scheduler,
                image_encoder=self.image_encoder,
                torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.pipe.unet_encoder = self.UNet_Encoder
        self.generator = torch.Generator(self.device).manual_seed(self.seed)                    


    def to_batch(self, tensors):
        batch = (
            torch.concatenate([tensor\
                               .to(device=self.device, dtype=self.weight_dtype)\
                               .unsqueeze(0)\
                               for tensor in tensors],
                               dim=0)
        )
        return batch

    @torch.inference_mode()
    def forward(self, input_data, single_cloth=True):
        
        if single_cloth:
            model_img = input_data["image"].to(device=self.device,
                                               dtype=self.weight_dtype).unsqueeze(0)
                        
            mask_img = self.to_pil(input_data["inpaint_mask"]*255).resize((self.width,self.height))
            
            pose_map = input_data["pose_map"].to(device=self.device,
                                                 dtype=self.weight_dtype).unsqueeze(0)
            pose_img_input = self.to_tensor(input_data["pose"].resize((self.width, self.height))).to(device=self.device,
                                                 dtype=self.weight_dtype).unsqueeze(0)

            cloth = input_data["cloth"].to(device=self.device,
                                           dtype=self.weight_dtype).unsqueeze(0)
            im_mask = input_data['im_mask'].to(device=self.device,
                                               dtype=self.weight_dtype).unsqueeze(0)
            human_img = input_data['image_human_try_on']
            pil_garment = input_data['cloth_orig']

            dense_pose_raw = self.to_tensor(input_data["dense_pose"].resize((self.width,self.height)))
            # r,g,b = dense_pose_raw[:,:,:]
            # dense_pose_raw[0,:,:], dense_pose_raw[1,:,:], dense_pose_raw[2,:,:] = r, g, b 

            
            dense_pose_img = dense_pose_raw.to(device=self.device,
                                               dtype=self.weight_dtype).unsqueeze(0)
            # TODO: в с оригинальным пайплайном различается только densepose. Быть аккуратным            

            path = "/usr/src/app/volume/tmp/idm_try_on/data/"
            def save_tensor(tensor, name):
                from torchvision.transforms.functional import to_pil_image
                to_pil_image(tensor.squeeze(0).cpu() ).convert('RGB').save(f"{path}/{name}")
            print(f"""
            # {type(dense_pose_img)=} {dense_pose_img.shape=}
            # """)
            save_tensor(cloth,"cloth.png")
            save_tensor(dense_pose_img,"pose_img.png")
            mask_img.save(f"{path}/mask.png")

            human_img.save(f"{path}/human_img.png")
            pil_garment.save(f"{path}/ip_adapter.png")           

            prompt_category = [input_data['category']]

            if input_data["cloth_desc"] is not None and len(input_data["cloth_desc"]) > 0:
                cloth_desc = [input_data['cloth_desc']]
            else:
                cloth_desc = None

        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if cloth_desc is not None:
            model_prompt = [f'model is wearing {desc}' for
                desc in cloth_desc]
            cloth_prompt = [f'a photo of {desc}' for
                desc in cloth_desc]
        else:
            category_text = {
                ImageCategory.DRESSES: 'a dress',
                ImageCategory.UPPER_BODY: 'an upper body garment',
                ImageCategory.LOWER_BODY: 'a lower body garment',
            }
            model_prompt = [f'a photo of a model wearing {category_text[category]}'
                    for category in prompt_category]
            cloth_prompt = [f'a photo of {category_text[category]}'
                    for category in prompt_category]

        logger.info(f"Setting text in try on pipeline: {model_prompt=}, {cloth_prompt=}")

        with torch.inference_mode():
            (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                model_prompt, # in original code it's str
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            (
            prompt_embeds_c,
            _,
            _,
            _,
            ) = self.pipe.encode_prompt(
                cloth_prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=negative_prompt,
            )

        generated_image = self.pipe(
                prompt_embeds=prompt_embeds.to(self.device, self.weight_dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.weight_dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.weight_dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.weight_dtype),
                num_inference_steps=self.num_inference_steps,
                generator=self.generator,
                strength = 1.0,
                pose_img = dense_pose_img.to(self.device, self.weight_dtype),
                text_embeds_cloth=prompt_embeds_c.to(self.device, self.weight_dtype),
                cloth = cloth.to(self.device, self.weight_dtype),
                mask_image=mask_img,
                image=human_img,
                height=self.height,
                width=self.width,
                ip_adapter_image=pil_garment.resize((self.width,self.height)),
                guidance_scale=self.guidance_scale,
            )[0][0]
        return generated_image
