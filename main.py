# %%
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from skimage.io import imread, imshow


from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
import cv2
from utils import plot_showcase


def extract_region(img, x, y, w, h):
    return img[y: y + h, x:x+w, ]


sc = SignClassifier(use_cached=True)


img_path = 'test_cases/Am_Rojo0002.jpg'
orig_img = cv2.imread(img_path)
orig_img = cv2.resize(orig_img, (500, 500))

# imshow(orig_img)

ss = SelectiveSearch()
regions = ss.propose_regions(orig_img)

numShowRects = 500
region_slices = []
region_labels = []
for i, rect in enumerate(regions):
    if (i < numShowRects):
        x, y, w, h = rect
        region_slices.append(extract_region(orig_img, x, y, w, h))
    else:
        break

region_labels = sc.predict(region_slices)
ann_img = np.array(orig_img, copy=True)
plt.imshow(ann_img)
plt.show()
for i, rect in enumerate(regions):
    # draw rectangle for region proposal till numShowRects
    if (i < numShowRects):
        x, y, w, h = rect
        if region_labels[i] == 1:
            cv2.rectangle(ann_img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            plt.imshow(ann_img)
    else:
        break

plt.imshow(ann_img)
plt.show()