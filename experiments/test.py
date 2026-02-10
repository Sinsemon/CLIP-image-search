import os
from sys import exit
from time import sleep
os.environ['HIP_VISIBLE_DEVICES'] = "1" # wichtig damit integrated gpu nicht erkannt wird!!
os.environ['AMD_LOG_LEVEL'] = "7"
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessorFast, CLIPVisionModel, CLIPTextModel, CLIPTokenizerFast
from torch import inference_mode
import torch


# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", return_dict=False)
# model_img = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14", return_dict=False)
# model_text = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", return_dict=False)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", return_dict=False)
model_img = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", return_dict=False)
model_text = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16", return_dict=False)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
# processor_img = CLIPImageProcessorFast.from_pretrained("openai/clip-vit-large-patch14")
# processor_text = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

model.to("cuda")
model_img.to("cuda")
model_text.to("cuda")

# %% Load and visualise the images
image_urls = [
    'http://images.cocodataset.org/val2014/COCO_val2014_000000159977.jpg', 
    'http://images.cocodataset.org/val2014/COCO_val2014_000000311295.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000457834.jpg', 
    'http://images.cocodataset.org/val2014/COCO_val2014_000000555472.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000174070.jpg',
    'http://images.cocodataset.org/val2014/COCO_val2014_000000460929.jpg'
    ]
classes = ['giraffe', 'zebra', 'elephant', 'teddybear', 'hotdog']
images = []
for url in image_urls:
    images.append(Image.open(requests.get(url, stream=True).raw))

with inference_mode():
    _text_inp = processor(text=classes, return_tensors="pt", padding=True)
    _text_inp.to("cuda")
    print("text tokenized")
    text_embeddings = model.get_text_features(**_text_inp)
    print("text embedded ", text_embeddings.shape)

    _img_input = processor(images=images, return_tensors="pt", padding=True)
    _img_input.to("cuda")
    print("images patchified")
    img_embeddings = model_img(**_img_input)
    img_emb_2 = model.get_image_features(**_img_input)
    print(type(img_embeddings))
    print(type(img_embeddings[0]))
    print(img_embeddings[0].shape)
    print(img_embeddings[0].dtype)
    print()
    print(img_emb_2.shape)
    print("images embedded")


# print("cpu")
# model_img.to("cpu")  # Irgendwas muss auf die CPU gemacht werden, damit sich Rocm oder HIP beendet... Sonst "Waiting for event 0000014DF796F6D0 to complete, current status 2"
# model_text.to("cpu")
# model.to("cpu")
# _text_inp.to("cpu")
# _img_input.to("cpu")
# # img_embeddings.to("cpu")
# text_embeddings.to("cpu")
# del model_img, model_text, model, _text_inp, _img_input, img_embeddings, text_embeddings
torch.cuda.synchronize()  # behebt ROCm/HIP-Shutdown-Deadlock
print("exit")


# ohne model(zeug)
# :5:commandqueue.cpp         :187 : 9801004089 us: [pid:4916 tid: 0x 6840] finish() called with batch size: 0, cpu_wait: 1, fence dirty: 0
# :5:commandqueue.cpp         :209 : 9801004182 us: [pid:4916 tid: 0x 6840] No HW event or batch size is less than 8, await command completion
# :4:commandqueue.cpp         :227 : 9801004223 us: [pid:4916 tid: 0x 6840] All commands finished for host queue : 000002A2910B1460

# :5:commandqueue.cpp         :187 : 9849277662 us: [pid:1176 tid: 0x20100] finish() called with batch size: 0, cpu_wait: 1, fence dirty: 0
# :5:commandqueue.cpp         :209 : 9849277755 us: [pid:1176 tid: 0x20100] No HW event or batch size is less than 8, await command completion
# :5:command.cpp              :357 : 9849277797 us: [pid:1176 tid: 0x20100] Command (InternalMarker) enqueued: 0000014DE362A9E0 to queue: 0000014DE3887EA0
# :5:command.cpp              :244 : 9849277853 us: [pid:1176 tid: 0x20100] Waiting for event 0000014DF796F6D0 to complete, current status 2


# :5:commandqueue.cpp         :187 : 9936031062 us: [pid:4472 tid: 0x 3188] finish() called with batch size: 0, cpu_wait: 1, fence dirty: 0
# :5:commandqueue.cpp         :209 : 9936031153 us: [pid:4472 tid: 0x 3188] No HW event or batch size is less than 8, await command completion
# :5:command.cpp              :357 : 9936031195 us: [pid:4472 tid: 0x 3188] Command (InternalMarker) enqueued: 0000028EE7606F00 to queue: 0000028EE7949F40
# :5:command.cpp              :244 : 9936031250 us: [pid:4472 tid: 0x 3188] Waiting for event 0000028F046EC210 to complete, current status 2

# :5:commandqueue.cpp         :187 : 10212478724 us: [pid:13784 tid: 0x10208] finish() called with batch size: 0, cpu_wait: 1, fence dirty: 0
# :5:commandqueue.cpp         :209 : 10212478821 us: [pid:13784 tid: 0x10208] No HW event or batch size is less than 8, await command completion
# :5:command.cpp              :357 : 10212478869 us: [pid:13784 tid: 0x10208] Command (InternalMarker) enqueued: 0000024718F3F320 to queue: 0000024719161310
# :5:command.cpp              :244 : 10212478925 us: [pid:13784 tid: 0x10208] Waiting for event 000002473632C720 to complete, current status 2