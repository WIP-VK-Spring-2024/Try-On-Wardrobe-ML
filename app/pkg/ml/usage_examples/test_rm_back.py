from app.pkg.ml.try_on.preprocessing.aggregator import ClothProcessor
from app.pkg.ml.try_on.preprocessing.cloth import BackgroundModels
from app.pkg.ml.try_on.preprocessing.cut_sam_pipeline.sam_points_strategies import PointsFormingSamStrategies

cp = ClothProcessor(BackgroundModels.SegFormerB3, True)

from PIL import Image
image = Image.open("/usr/src/app/data/upper/b_t-shirt-2.png")
im_bytes = cp.bytes_converter.image_to_bytes(image)

im_no_back_bytes = cp.consistent_forward(
    image_bytes=im_bytes,
    point_sam_strategy=PointsFormingSamStrategies.strategy_3
    )['cloth']

cp.bytes_converter.bytes_to_image(im_no_back_bytes).save("/usr/src/app/data/etc/1_3_bria.png")

# im_no_back = cp.model_background(
#     cloth_im=image,
#     point_sam_strategy=PointsFormingSamStrategies.strategy_3)
# im_no_back.save("/usr/src/app/data/etc/1_3.png")
