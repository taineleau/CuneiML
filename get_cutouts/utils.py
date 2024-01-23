from shapely.geometry import Polygon
from PIL import ImageDraw as draw
import json
from PIL import Image
import numpy as np

def compute_iou_from_contours(poly1, poly2) -> float:
    """
    Get the Intersection-over-Union value between two contours.
    :param poly1: The first contour used to compute the
                  Intersection-over-Union.
    :param poly2: The second contour used to compute the
                  Intersection-over-Union.
    :return: The computed Intersection-over-Union value between the two
             contours.
    """
    if not poly1 or not poly2:
        return -1
    intersection = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - intersection
    return intersection / union if union != 0 else 1

def read_cc_based(path, t=0.01):
    """
        path: json path to the stored boxes coors
        t:    the threshold to filter out small boxes
    """
    boxes = json.load(open(path))
    
    # read the image to get the size of images, no need to read the img again if you have it already!
    img = Image.open(path.replace(".json", ".jpg"))
    h, w = img.size
    full_area = h*w
    
    filter_boxes = []
    for box in boxes:
        poly = Polygon(box)
        if poly.area / full_area > t:
            filter_boxes.append(box)
    return filter_boxes
    
def read_peak_based(path, t=0.12, viz=False):
    
    # read the image to get the size of images, no need to read the img again if you have it already!
    img = Image.open("/trunk/shared/cuneiform/full_data/images/" + path.replace(".json", ".jpg").split('/')[-1])
    h, w = img.size

    res = json.load(open(path))
    
    # the split points (either vertical or horizontal)
    # so these lines segment the orginal images into different regions
    splits =  {
        "v_split": [0] + sorted(res['col_res']) + [h], 
        "h_split": [0] + sorted(res['row_res']) + [w]
    }
    # print(splits)
    boxes = []
    max_v = 0
    for i in range(1, len(splits['v_split'])):
        for j in range(1, len(splits['h_split'])):
            x0 = splits['v_split'][i - 1]
            y0 = splits['h_split'][j - 1]
            x1 = splits['v_split'][i]
            y1 = splits['h_split'][j]
            # print((x0, y0, x1, y1))
            patch = img.crop((x0, y0, x1, y1))
            # display(patch)
            mean_v =  np.array(patch).mean()
            boxes.append([(x0, y0, x1, y1), mean_v])
            # 
            if mean_v > max_v:
                max_v = mean_v
            # print(np.array(patch).mean())
            
    # filter those empty box by the mean_v

    filter_boxes = []
    for b, v in boxes:
        if v > max_v * t:
            if viz:
                display(img.crop(b))
            filter_boxes.append(b)
    return filter_boxes

def get_front_polygon(example):
    areas = {}
    for idx, coords in enumerate(example):
        polygon = Polygon(coords)
        areas[idx] = polygon.area
    areas = dict(sorted(areas.items(), key=lambda item: item[1], reverse=True))
    if(len(areas)==1):
        return example[0]
    top_two_polygons = (example[list(areas.keys())[0]], example[list(areas.keys())[1]])

    min_y = float('inf')
    front_poly_idx = -1
    for idx, poly in enumerate(top_two_polygons):
        for coord in poly:
            if min_y > coord[1]:
                min_y = coord[1]
                front_poly_idx = idx
    return top_two_polygons[front_poly_idx]

def getFrontCutoutForSA(masks):
    frontMask = None
    if len(masks) == 0:
        return frontMask
    elif len(masks) == 1:
        frontMask = masks[0]['bbox']
    elif len(masks) > 2:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            #masks[0] is background, choose from 1 or 2
            if masks[1]['area'] > masks[2]['area']*1.5: #masks[1] is much bigger than masks[2]
                frontMask = masks[1]['bbox']
            elif masks[1]['bbox'][1] < masks[2]['bbox'][1]:  # ycoordinate of front mask will be smaller
                frontMask = masks[1]['bbox']
            else:
                frontMask = masks[2]['bbox']
        else:
            frontMask = masks[0]['bbox']
    else:
        if masks[0]['bbox'][0] <=5 and masks[0]['bbox'][1] <=5:
            frontMask = masks[1]['bbox']
        else:
            frontMask = masks[0]['bbox']
            
    return frontMask

def convert_peak_to_coords(peak_example):
    coords_example = []
    for coords in peak_example:
        x_0 = coords[0]
        y_0 = coords[1]
        x_1 = coords[2]
        y_1 = coords[3]
        curr = [[x_0,y_0],[x_0,y_1],[x_1,y_1],[x_1,y_0]]
        coords_example.append(curr)
    return coords_example
        
def convert_sa_to_coords(sa_front):
    x,y,w,h = sa_front
    return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]

def resizeCCExample(cc_example):
    cc_np = np.array(cc_example)
    while np.any(cc_np>2000):
        cc_np = cc_np/2
    return cc_np.tolist()