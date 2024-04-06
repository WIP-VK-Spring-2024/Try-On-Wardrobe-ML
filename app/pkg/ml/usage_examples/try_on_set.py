from PIL import Image

from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor
from app.pkg.ml.try_on.lady_vton import LadyVtonAggregator

from app.pkg.models.app.image_category import ImageCategory

from app.pkg.ml.buffer_converters import BytesConverter
bc = BytesConverter()

upper_fp = "/usr/src/app/data/example/t-shirt.png"
lower_fp = "/usr/src/app/data/example/white_pants.png"
human_fp = "/usr/src/app/data/example/woman_resize.png"


user_image = Image.open(human_fp)
upper_cloth = Image.open(upper_fp)
lower_cloth = Image.open(lower_fp)


human_model = HumanProcessor()
cloth_model = ClothProcessor()
try_on_model = LadyVtonAggregator()



processed_user = human_model.process(user_image)
# print(processed_user.keys())
bc.bytes_to_image(processed_user['pose']).save("/usr/src/app/volume/tmp/1_pose.jpg")
bc.bytes_to_image(processed_user['parse_orig']).save("/usr/src/app/volume/tmp/1_parse.png")
bc.bytes_to_image(processed_user['image_human_orig']).save("/usr/src/app/volume/tmp/1_orig.jpg")


no_background_image_upper = cloth_model.model_background(upper_cloth)
no_background_image_bytes = bc.image_to_bytes(no_background_image_upper)
upper_dict = {}
upper_dict["cloth"] = no_background_image_bytes   
upper_dict['category'] = ImageCategory.UPPER_BODY


no_background_image_lower = cloth_model.model_background(lower_cloth)
no_background_image_bytes = bc.image_to_bytes(no_background_image_lower)
lower_dict = {}
lower_dict["cloth"] = no_background_image_bytes   
lower_dict['category'] = ImageCategory.LOWER_BODY

no_background_image_upper.save("/usr/src/app/volume/tmp/upper.png")
no_background_image_lower.save("/usr/src/app/volume/tmp/lower.png")

# processed_cloth = result

# Try on
# processed_user.update(
# {
#     "category": "upper_body",
#     "cloth": no_background_image_bytes,
# }
# )

try_on = try_on_model.try_on_set(human=processed_user,
                                 clothes=[upper_dict,lower_dict])

bc.bytes_to_image(try_on).save("/usr/src/app/volume/tmp/try_on7.jpg")
#  python3 -m app.pkg.ml.usage_examples.try_on 
