# import os
# # os.environ['AMD_LOG_LEVEL'] = "7"
# os.environ['HIP_VISIBLE_DEVICES'] = "1" # wichtig damit integrated gpu nicht erkannt wird!!
# import torch
# import clip
# from PIL import Image
# import requests

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
# print(clip.available_models())
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open(requests.get("http://images.cocodataset.org/val2014/COCO_val2014_000000159977.jpg", stream=True).raw)).unsqueeze(0).to(device)
# print("hallo 0")
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print("hallo 1")

# with torch.no_grad():
#     print("hallo 1,5")
#     text_features = model.encode_text(text)
#     print("hallo 2")
#     image_features = model.encode_image(image)
#     print("hallo 3")
    
#     logits_per_image, logits_per_text = model(image, text)
#     print("hallo 4")
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

import torch
import torch.nn.functional as F

input1 = torch.randn(10, 512)
input2 = torch.randn(1, 512)
output = F.cosine_similarity(input1, input2)
print(output)