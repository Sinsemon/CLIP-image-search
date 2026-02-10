from const import DEVICE

from transformers import CLIPModel, CLIPProcessor
from torch import inference_mode, cuda
from PIL import Image



class Clip:  # TODO vlt von Torch erben?

    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", return_dict=False)
        self.model.to(DEVICE)
    
    def __del__(self):
        cuda.synchronize() # ROCm/HIP-Shutdown-Deadlock

    def embed_text(self, text:list[str]):
        _text_inp = self.processor(text=text, return_tensors="pt", padding=True)
        _text_inp.to(DEVICE)
        return self.model.get_text_features(**_text_inp)

    def embed_images(self, images:list[Image.Image]):
        _img_input = self.processor(images=images, return_tensors="pt", padding=True)
        _img_input.to(DEVICE)
        return self.model.get_image_features(**_img_input)
