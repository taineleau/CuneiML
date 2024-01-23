from shapely.geometry import Polygon
from PIL import ImageDraw as draw
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
random.seed(41)
import tqdm
from ensembleSegmentationUtils import *
import os

with open('/trunk/shared/cuneiform/full_data/all_ids.json', 'r') as f:
    all_ids = json.load(f)
    
with open('/trunk/shared/cuneiform/CuneiformDating/image_classification/segmentation/code/temp_results/iou_segmentation.json', 'r') as f:
    iou_info = json.load(f)
   
for idx, pid in tqdm.tqdm(enumerate(all_ids)):
    cc_path = "/trunk2/datasets/cuneiform/segmentation/seg_viz_July05/P" + str(pid).zfill(6)+ ".json"
    if os.path.exists(cc_path):
        cc_example = read_cc_based(cc_path)
        cc_example = resizeCCExample(cc_example)
        cc_front = get_front_polygon(cc_example)
        poly_cc = Polygon(cc_front)
    else:
        poly_cc = None

    peak_path="/trunk2/datasets/cuneiform/segmentation/seg_peak_base_July06/P" + str(pid).zfill(6)+ ".json"
    if os.path.exists(peak_path):
        peak_example = read_peak_based(peak_path)
        peak_front = get_front_polygon(convert_peak_to_coords(peak_example))
        poly_peak = Polygon(peak_front)
    else:
        poly_peak = None


    segmentAnything_path = "/trunk/shared/cuneiform/full_data/segmented_mask_info_compressed/P" + str(pid).zfill(6)+ ".pkl"
    if os.path.exists(segmentAnything_path):
        with open(segmentAnything_path, 'rb') as f:
            all_masks = pickle.load(f)

        sa_front = getFrontCutoutForSA(all_masks)
        sa_front = convert_sa_to_coords(sa_front)
        poly_sa = Polygon(sa_front)
    else:
        poly_sa = None

    iou_sa_cc = compute_iou_from_contours(poly_cc,poly_sa)
    iou_sa_peak = compute_iou_from_contours(poly_peak,poly_sa)
    iou_peak_cc = compute_iou_from_contours(poly_cc,poly_peak)

    iou_info[pid] = {"iou_sa_cc":iou_sa_cc, "iou_sa_peak":iou_sa_peak, "iou_peak_cc":iou_peak_cc}
    
    if idx%1000==0 or idx == len(all_ids)-1:
        with open("/trunk/shared/cuneiform/CuneiformDating/image_classification/segmentation/code/temp_results/iou_segmentation.json", "w") as f:
            json.dump(iou_info,f)

        
  