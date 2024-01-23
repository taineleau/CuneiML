from sklearn.cluster import KMeans
import numpy as np
from scipy.signal import find_peaks
from PIL import Image

def peak_base(img_url, viz=False):
    img = Image.open(img_url)

    arr_grey = np.array(img.convert("L"))

    row_sum = np.sum(arr_grey, axis=1)
    col_sum = np.sum(arr_grey, axis=0)
    x = col_sum.max() - col_sum
    peaks, _ = find_peaks(x)
    
    filter_peaks = []
    for p in peaks:
        # print(p, x.max() * 0.9)
        if x[p] > x.max() * 0.9:
            filter_peaks.append(p)
    # print(filter_peaks)
    kmeans = KMeans(n_clusters=4).fit(np.array(filter_peaks).reshape(-1, 1))
    res = {}
    for i in range(4):
        res[i] = []
    for l, p in zip(kmeans.labels_, filter_peaks):
        res[l].append(p)

    from PIL import ImageDraw

    w, h = img.size
    print(w, h)
    base = img.convert('RGB')
    draw = ImageDraw.Draw(base)

    for i in range(4):
        line = int(np.mean(res[i]))
        print(line)
        draw.line((line, 0, line, h), fill='red', width=3)
        
    # draw.line((2, 3, 500, 555), fill='red', width=10)


    x = row_sum.max() - row_sum
    peaks, _ = find_peaks(x)
    filter_peaks = []
    for p in peaks:
        # print(p, x.max() * 0.9)
        if x[p] > x.max() * 0.9:
            filter_peaks.append(p)
    # print(filter_peaks)

    kmeans = KMeans(n_clusters=5).fit(np.array(filter_peaks).reshape(-1, 1))
    res = {}
    
    for i in range(5):
        res[i] = []
    for l, p in zip(kmeans.labels_, filter_peaks):
        res[l].append(p)
    for l, p in zip(kmeans.labels_, filter_peaks):
        res[l].append(p)

    for i in range(5):
        line = int(np.mean(res[i]))
        print(line)
        draw.line((0, line, w, line), fill='red', width=3)
        
    if viz:
        display(base)

    