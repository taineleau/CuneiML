import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import tqdm
import json
import os
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import pickle

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# %%
device = "cuda:3" if torch.cuda.is_available() else "cpu"

# %%
with open("all_ids.json", 'r') as f:
    all_ids = json.load(f)
len(all_ids)

# %%
sam_checkpoint = "../image_classification/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=5,
    pred_iou_thresh=0.94,
    stability_score_thresh=0.90,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=10000,  # Requires open-cv to run post-processing
)

# %%
max_dim = 2000
min_dim = 600

def getFrontCutout(masks,image):
    frontMask = None
    if len(masks) == 0:
        return frontMask
    elif len(masks) == 1:
        frontMask = masks[0]
    else:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            frontMask = masks[1]
        else:
            frontMask = masks[0]
    x,y,w,h = frontMask['bbox']
    x,y,w,h = int(x), int(y), int(w), int(h)
    cutout = image[y:y+h, x:x+w]
    return cutout

def resizeImage(image):
    
    while image.shape[0]/max_dim > 1 or image.shape[1]/max_dim > 1:
        dim = (int(image.shape[0]/2), int(image.shape[1]/2))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# %%
problem_images = []

# %%
for idx, pid in tqdm.tqdm(enumerate(all_ids[53876:])):
    try:
        image_name = "P"+ str(pid).zfill(6)+".jpg"
        image = cv2.imread("images/"+image_name)
        newFilePath = "./segmented_mask_info_compressed/P" + str(pid).zfill(6) +".pkl"
        width = image.shape[1]
        height = image.shape[0]
        
        #if low resolution image, keep as it is
        if height<=min_dim and width<=min_dim:
            cv2.imwrite("segmented_images/"+image_name, image)
            continue
            
        #if very high resolution image, then resize
        if height>max_dim or width>max_dim:
            image = resizeImage(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        masks = sorted(masks, key = lambda d: d['area'], reverse = True)
        topFive = masks[:5]
        
        with open(newFilePath, 'wb') as f:
            pickle.dump(topFive,f)
    
        cutout = getFrontCutout(masks, image)
        cutout = cv2.cvtColor(cutout, cv2. COLOR_BGR2RGB)
        cv2.imwrite("segmented_images/"+image_name, cutout)

    except:
        print("Exception occured", pid)
        problem_images.append(pid)