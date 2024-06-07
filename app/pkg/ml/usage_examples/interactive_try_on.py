import torch
from PIL import Image
import numpy as np

from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.dense_pose import DensePoseEstimation

from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.models_aggregator import TryOnAggregator, TryOnModels
# from app.pkg.ml.try_on.postprocessing.fix_face import FaceFixer

from app.pkg.ml.buffer_converters import BytesConverter
from app.pkg.models.app.image_category import ImageCategory

bc = BytesConverter()

human_model = HumanProcessor()
cloth_model = ClothProcessor()
try_on_model = TryOnAggregator(TryOnModels.IDM_VTON)


cloth_fp = "/usr/src/app/data/upper/t-shirt-miami.png"
lower_cloth_fp = "/usr/src/app/data/lower/b_shorts.png"
human_fp = "/usr/src/app/data/human/jennifer_lourence.png"

user_image = Image.open(human_fp)  # .resize((512,512))

def get_cloth_from_fp(path, category:ImageCategory, desc):
    cloth_image = Image.open(path)
    no_background_image = cloth_model.model_background(cloth_image)
    no_background_image_bytes = bc.image_to_bytes(no_background_image)

    return {
        "cloth": no_background_image_bytes,
        "category": category,
        "cloth_desc":desc,
    }
    
while True:
    try:
        human_fp = input("User image fp: ") #"/usr/src/app/data/human/jennifer_lourence.png"

        user_image = Image.open(human_fp)  # .resize((512,512))

        upper_cloth_fp = input("Upper cloth fp: ")
        upper_cloth_desc = input("Upper cloth desc: ") #"t-shirt"
        
        lower_cloth_fp = input("Lower cloth fp: ")
        lower_cloth_desc = input("Lower cloth desc: ") 
        
        change_model_params = bool(int(input("change_model_params?")))
        if change_model_params:
            stop_flag = bool(int(input("stop?")))
            while not stop_flag:
                param_name = input("param_name: ")
                param_value = int(input("param_value: "))

                setattr(try_on_model.model, param_name, param_value)
                
                stop_flag = bool(int(input("stop?")))
        print("Starting try on...")
        upper_cloth_dict = get_cloth_from_fp(upper_cloth_fp, ImageCategory.UPPER_BODY, upper_cloth_desc)
        lower_cloth_dict = get_cloth_from_fp(lower_cloth_fp, ImageCategory.LOWER_BODY, lower_cloth_desc)
        # Try on
        processed_user = human_model.process(user_image)

        processed_user.update(upper_cloth_dict)


        save_index = np.random.randint(9, 100000000)
        save_fp = f"/usr/src/app/volume/tmp/idm_try_on/single_{save_index}.png"
        try_on = try_on_model(processed_user)
        bc.bytes_to_image(try_on).save(save_fp)
        print(f"End of single cloth pipeline. Saving at: {save_fp}")
        print("Starting outfit try on")


        processed_user = human_model.process(user_image)

        save_index = np.random.randint(9, 100000000)
        save_fp = f"/usr/src/app/volume/tmp/idm_try_on/single_{save_index}.png"
        try_on = try_on_model.try_on_set(processed_user, clothes=[ upper_cloth_dict, lower_cloth_dict] )
        bc.bytes_to_image(try_on).save(save_fp)
        print(f"End of set of clothes pipeline. Saving at: {save_fp}")
  


    except Exception as e:
        print(e)












# python3 -m app.pkg.ml.usage_examples.test
