import transformers

from transformers import CLIPProcessor, CLIPModel


class Tagger:
    def __init__(self, dataset_path, weights_path, device='cpu'):
        self.dataset_path = dataset_path
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def get_tags(self, user_id, image_id, tags,):
        """
        Selects tags from given tags through image
        with image_id and user_id
        """
        inputs = self.clip_processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)

