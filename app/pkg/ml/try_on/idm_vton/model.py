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
# from utils_mask import get_mask_location
# import apply_net
# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose # TODO: replace to existing path
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation

# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose # TODO: replace to existing path
# from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from app.pkg.ml.try_on.idm_vton.IDM_VTON.src.unet_hacked_tryon import UNet2DConditionModel

from app.pkg.models.app.image_category import ImageCategory
from app.pkg.settings import settings


base_path = 'yisol/IDM-VTON'



class IDM_VTON(torch.nn.Module):
    """
    Virtual try on model implementation    
    """
    def __init__(self, num_inference_steps=20):
        super(IDM_VTON, self).__init__()
        self.weight_dtype = torch.float16#torch.float32
        # self.data_prepr = LadyVtonInputPreprocessor()
        self.device = "cuda:0"
        # self.dataset = "dresscode"
        # self.pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-inpainting"
        # self.enable_xformers_memory_efficient_attention = True
        self.seed = 42
        # self.num_vstar = 16  # Number of predicted v* images to use
        self.guidance_scale = 2 # 7.5
        self.num_inference_steps = num_inference_steps
        self.setup_models()
        self.WEIGHTS_PATH = settings.ML.WEIGHTS_PATH
        self.DENSE_POSE_WEIGHTS_PATH = f"{self.WEIGHTS_PATH}/dense_pose.pkl"


    def setup_models(self):
        print('start initing models')
        self.unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        )
        print('start initing models 1')

        self.unet.requires_grad_(False)
        self.tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        print('start initing models 11')

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        print('start initing models 111')

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        )
        print('start initing models 2')

        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=self.weight_dtype,
            )
        self.vae = AutoencoderKL.from_pretrained(base_path,
                                            subfolder="vae",
                                            torch_dtype=self.weight_dtype,
        )

        print('start initing models 3')

        # "stabilityai/stable-diffusion-xl-base-1.0",
        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=self.weight_dtype,
        )

        # self.parsing_model = Parsing(0)
        # self.openpose_model = OpenPose(0)

        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)
        self.tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )
        self.to_tensor = transforms.ToTensor()

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
        
        print("Gotcha state 1")
        if single_cloth:
            model_img = input_data["image"].to(device=self.device,
                                               dtype=self.weight_dtype).unsqueeze(0)
            mask_img = input_data["inpaint_mask"].to(device=self.device,
                                                     dtype=self.weight_dtype).unsqueeze(0)

            pose_map = input_data["pose_map"].to(device=self.device,
                                                 dtype=self.weight_dtype).unsqueeze(0)
            pose_img_input = self.to_tensor(input_data["pose"].resize((768, 1024))).to(device=self.device,
                                                 dtype=self.weight_dtype).unsqueeze(0)

            cloth = input_data["cloth"].to(device=self.device,
                                           dtype=self.weight_dtype).unsqueeze(0)
            im_mask = input_data['im_mask'].to(device=self.device,
                                               dtype=self.weight_dtype).unsqueeze(0)

            prompt_category = [input_data['category']]

            if input_data["cloth_desc"] is not None and len(input_data["cloth_desc"]) > 0:
                cloth_desc = [input_data['cloth_desc']]
            else:
                cloth_desc = None
        # prompt = "model is wearing " + garment_des
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        print("Gotcha state 2")

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

        print("Gotcha state 3")

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

            # args = apply_net.create_argument_parser().parse_args((
            #     'show',
            #     "app/pkg/ml/try_on/preprocessing/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml", # здесь мб с точки должно начинаться
            #     self.DENSE_POSE_WEIGHTS_PATH,
            #     'dp_segm',
            #     '-v',
            #     '--opts',
            #     'MODEL.DEVICE',
            #     'cuda'))
            # verbosity = getattr(args, "verbosity", None)
            
            # human_img_arg = _apply_exif_orientation(model_img) # TODO: .resize((384,512))
            # human_img_arg =  human_img_arg# convert_PIL_to_numpy(human_img_arg, format="BGR")

            # pose_img = args.func(args, human_img_arg)
            # pose_img = pose_img[:,:,::-1]    
            # pose_img = Image.fromarray(pose_img).resize((768,1024))
            pose_img = pose_img_input
            print("Lets go")
            print(
                f"""
                prompt_embeds={prompt_embeds.shape},
                negative_prompt_embeds={negative_prompt_embeds.shape},
                pooled_prompt_embeds={pooled_prompt_embeds.shape},
                negative_pooled_prompt_embeds={negative_pooled_prompt_embeds.shape},
                pose_img = {pose_img.shape},
                text_embeds_cloth={prompt_embeds_c.shape},
                # cloth = cloth,
                mask_image={mask_img.shape},
                image=model_img,
                ip_adapter_image = cloth, # в оригинальном пайплайне это pil image
                """


            )
            generated_image = self.pipe(
                prompt_embeds=prompt_embeds.to(self.device, self.weight_dtype),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device, self.weight_dtype),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, self.weight_dtype),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, self.weight_dtype),
                num_inference_steps=self.num_inference_steps,
                generator=self.generator,
                strength = 1.0,
                pose_img = pose_img.to(self.device, self.weight_dtype),
                text_embeds_cloth=prompt_embeds_c.to(self.device, self.weight_dtype),
                cloth = cloth,
                mask_image=mask_img,
                image=model_img,
                height=1024,
                width=768,
                ip_adapter_image = cloth, # в оригинальном пайплайне это pil image
                guidance_scale=self.guidance_scale,
            )[0]
          
        return generated_image
