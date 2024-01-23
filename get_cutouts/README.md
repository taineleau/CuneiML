## Cutouts


basic requirement:

```
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

#### Algorithm 1: Connected component based

`CCbased.py` We first convert the images into black and white where the background is black. This is a classical rule-based segmentation that clusters the adjacent pixels with the same color. TWe use OpenCV’s implementation cv.connenctedComponents()

#### Algorithm 2: Peak-based segmentation

`peakbased.py` We do a row/column sum over the image, and find the peak using spicy's `find_peaks()`.


#### Alogirthm 3: Segment Anything (SAM)


`SAMbased.py` This is the state-of-the-art general segmentation algorithm using neural networks. We uses the oﬀicial toolkit (https://github.com/facebookresearch/segment-anything) with default model weights (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) to obtain cutouts.



### Ensemble

After extacting faces using three algorithms, we compute the iou (in `ensemble.py`) to filter good cutouts from the ensemble of the three algorithms.