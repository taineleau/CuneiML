
### connected component
import cv2 as cv, cv2
import numpy as np
import imutils
import os
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import json

def cut_piece(img_url='test.png', viz=False):
    
    file_name = img_url.split("/")[-1]
    save_name = f"/trunk2/datasets/cuneiform/segmentation/seg_viz_July05/{file_name}"
    if os.path.exists(save_name):
        if viz:
            display(Image.open(save_name))
        return
    
    img = cv.imread(img_url)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # gray = cv.bitwise_not(gray)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)

    num_labels, labels_im = cv.connectedComponents(cv.bitwise_not(sure_bg))

    # plt.imshow(labels_im)


    boxes = []
    for label in np.unique(labels_im):
    # if the label is zero; it is the background so ignore
        if label == 0:
            continue
        # get a mask for each segement
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels_im == label] = 255
        # detect contours in the mask and grab the largest one
        # plt.imshow(mask)
        # plt.show()
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # getting a bounding box around
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
        
    from PIL import ImageDraw
    base = Image.open(img_url).convert("RGB")
    draw = ImageDraw.Draw(base)
    h, w = base.size
    full_area = h*w
    for box in boxes:
        poly = Polygon(box)
        if poly.area / full_area > 0.01:
            draw.rectangle((box[0][0], box[0][1], box[2][0], box[2][1]), outline='red')
    if viz:
        display(base)
    
    file_name = img_url.split("/")[-1]
    base.save(f"/trunk2/datasets/cuneiform/segmentation/seg_viz_July05/{file_name}")
    json_name = file_name.split(".")[0] + ".json"
    boxes = [x.tolist() for x in boxes]
    json.dump(boxes, open(f"/trunk2/datasets/cuneiform/segmentation/seg_viz_July05/{json_name}", "w"))
