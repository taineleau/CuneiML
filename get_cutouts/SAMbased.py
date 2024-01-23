import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
import tqdm
import json

# !{sys.executable} -m pip install opencv-python matplotlib
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image
import random
import pickle

import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

random.seed(41)

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
    
def getFrontCutout(masks,image):
    frontMask = None
    if len(masks) == 0:
        return frontMask
    elif len(masks) == 1:
        frontMask = masks[0]
    elif len(masks) == 2:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            #masks[0] is background, return 1
            frontMask = masks[1]
        else:
            #choose between 0 or 1
            if masks[0]['area'] > masks[1]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[0]
            elif masks[0]['bbox'][1] < masks[1]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[0]
            else:
                frontMask = masks[1]
    else:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            #masks[0] is background, choose from 1 or 2
            if masks[1]['area'] > masks[2]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[1]
            elif masks[1]['bbox'][1] < masks[2]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[1]
            else:
                frontMask = masks[2]

        else:
            #choose between 0 or 1
            if masks[0]['area'] > masks[1]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[0]
            elif masks[0]['bbox'][1] < masks[1]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[0]
            else:
                frontMask = masks[1]
                
    x,y,w,h = frontMask['bbox']
    x,y,w,h = int(x), int(y), int(w), int(h)
    cutout = image[y:y+h, x:x+w]
    return cutout

def resizeImage(image, max_dim):
    
    while image.shape[0]/max_dim > 1 or image.shape[1]/max_dim > 1:
        dim = (int(image.shape[1]/2), int(image.shape[0]/2))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image 


device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(41)

max_dim = 2000
min_dim = 600

sam_checkpoint = "/trunk/shared/cuneiform/CuneiformDating/image_classification/segmentation/sam_vit_h_4b8939.pth"
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

ids_path = "/trunk/shared/cuneiform/full_data/all_ids.json'

with open(ids_path, 'r') as f:
    all_ids = json.load(f)

print("Total images to segment:", len(all_ids))

image_anno = json.load(open("/trunk2/datasets/cuneiform/image_anno.json", 'r'))


for pid in tqdm.tqdm(all_ids):
    try:

        image_path = "/trunk/shared/cuneiform/full_data/images/"+ "P"+ str(pid).zfill(6)+".jpg"
        masks_filepath = "/trunk/shared/cuneiform/full_data/segmented_mask_info_compressed/P" + str(pid).zfill(6) +".pkl"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width = image.shape[1]
        height = image.shape[0]

        if not ("RGB" in image_anno[pid].keys() and image_anno[pid]["RGB"]): #if non RGB image, no segmentation
            cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+image_name, image)
            continue
            
#         #if low resolution, use original image
#         if height<=min_dim and width<=min_dim:
#             cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+image_name, image)
#             continue

        #if very high resolution image, then resize and save the resized image
        if height>max_dim or width>max_dim:
            image = resizeImage(image, max_dim)
            cv2.imwrite("/trunk/shared/cuneiform/full_data/images/"+ "P"+ str(pid).zfill(6)+".jpg", image)

        #adjust contrast for better segmentation
        enhanced_image = image.copy()
        black_mask = image < 30
        enhanced_image[black_mask] = 0

        masks = mask_generator.generate(enhanced_image)
        masks = sorted(masks, key = lambda d: d['area'], reverse = True)
        topFive = masks[:5]

        with open(masks_filepath, 'wb') as f:
            pickle.dump(topFive,f)

        cutout = getFrontCutout(topFive, image)
        cutout = cv2.cvtColor(cutout, cv2. COLOR_BGR2RGB)
        cv2.imwrite("/trunk/shared/cuneiform/full_data/segmented_images/"+ "P"+ str(pid).zfill(6)+".jpg", cutout)

    except:
        print(pid)
        