from app.pkg.ml.try_on.preprocessing.aggregator import HumanProcessor

from app.pkg.ml.buffer_converters import BytesConverter
from PIL import Image
import numpy as np
bc = BytesConverter()


hp = HumanProcessor()

human_path  = "data/human/jennifer_lourence.png"
human_pil = Image.open(human_path)
human_bytes = bc.image_to_bytes(human_pil)
res = hp.consistent_forward(human_bytes)
print()
dense = bc.bytes_to_image(res['dense_pose'])
dense.save("/usr/src/app/volume/tmp_images/2.png")

dense_np = np.array(dense)[:, :, ::-1]
print(dense_np.shape)

Image.fromarray(dense_np).save("/usr/src/app/volume/tmp_images/3.png")


# python3 -m app.pkg.ml.usage_examples.test_dense_pose