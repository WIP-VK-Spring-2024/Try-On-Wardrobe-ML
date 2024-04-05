from PIL import Image

from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator
from app.pkg.ml.try_on.postprocessing.fix_face import FaceFixer

from app.pkg.ml.buffer_converters import BytesConverter
bc = BytesConverter()

cloth_fp = "/usr/src/app/data/example/t-shirt.png"
human_fp = "/usr/src/app/data/example/human_shirt.png"


user_image = Image.open(human_fp)
cloth_image = Image.open(cloth_fp)


human_model = HumanProcessor()
#cloth_model = ClothProcessor()
# try_on_model = LadyVtonAggregator()



processed_user = human_model.process(user_image)
print(processed_user.keys())
bc.bytes_to_image(processed_user['pose']).save("/usr/src/app/volume/tmp/1_pose.jpg")
bc.bytes_to_image(processed_user['parse_orig']).save("/usr/src/app/volume/tmp/1_parse.png")
bc.bytes_to_image(processed_user['image_human_orig']).save("/usr/src/app/volume/tmp/1_orig.jpg")


# no_background_image = cloth_model.model_background(cloth_image)
# no_background_image_bytes = bc.image_to_bytes(no_background_image)
# result = {}
# result["cloth"] = no_background_image_bytes   
# processed_cloth = result

# # Try on
# processed_user.update(
# {
#     "category": "upper_body",
#     "cloth": no_background_image_bytes,
# }
# )

# try_on = try_on_model(processed_user)

# bc.bytes_to_image(try_on).save("/usr/src/app/volume/tmp/try_on4.jpg")
# #  python3 -m app.pkg.ml.usage_examples.try_on 