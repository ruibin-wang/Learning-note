# CLIP image-text code


## Introduction

* The paper (**Learning Transferable Visual Models From Natural Language Supervision**) of this method can be found at: https://arxiv.org/pdf/2103.00020.pdf

* official link can be found at: https://openai.com/blog/clip/


## Code
```python
## first, install the dependency
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git


## configuration
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

## 
image_path = "220px-Queen_Elizabeth_II_of_New_Zealand.jpg"   ## add the path of image
text_list = ["white dress", "woman", "lady", "old lady", "queen", "British"]  ## pre-define according to the image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = clip.tokenize(text_list).to(device)


with torch.no_grad():
    image_features = model.encode_image(image)
    text_feature = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()


print("Label probes:", probs)


## output

Label probes: [[6.1566749e-04 3.6723554e-02 6.4185143e-02 3.9823200e-03 6.7976791e-01
  2.1472540e-01]]


```




