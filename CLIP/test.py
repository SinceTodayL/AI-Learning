import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
image = preprocess(Image.open('dog2.jpg')).unsqueeze(0).to(device)

texts = ["a big white dog", "a small orange dog", "a white cat", "an ugly cat"]
texts_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    image_feature = model.encode_image(image)
    print(image_feature.shape)
    text_feature = model.encode_text(texts_tokens)
    print(text_feature.shape)

image_feature /= image_feature.norm(dim=-1, keepdim=True)
text_feature /= text_feature.norm(dim=-1, keepdim=True)

similar = (image_feature @ text_feature.T).squeeze(0)

best_match = similar.argmax().item()
print(f"\nbest match: {texts[best_match]}")
print("All Score: ")
for i, (text, score) in enumerate(zip(texts, similar)):
    print(f"  {text:<20s}: {score:.4f}")
