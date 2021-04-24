
# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from math import sqrt

from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
import cv2

from image_utils import turnoff_irrelevant


class SignDetector:

    def __init__(self, pretrained=True):
        self.sign_classifier = SignClassifier(use_cached=pretrained)
        self.iou_th = 0.01
        self.prob_th = 0.1

    def detect_signs(self, orig_img):

        imgt = turnoff_irrelevant(orig_img)
        img = imgt.copy()
        
        #exit()
        ss = SelectiveSearch()
        regions, canny_img = ss.propose_regions_countour(img)
        print(f'Number of proposed regions = {len(regions)}')

        region_slices = []
        region_probs = []
        for i, rect in enumerate(regions):
            x, y, w, h = rect
            region_slices.append(self.extract_region(img, x, y, w, h))

        region_probs = self.sign_classifier.predict(
            region_slices)

        n_probs = len(region_probs)
        self.prob_th = sorted(region_probs, reverse=True)[round(n_probs*0.1)]
        print(f'confidence threshold = {self.prob_th}')

        candidates = [(region, prob, i) for i, (region, prob) in enumerate(
            zip(regions, region_probs)) if prob >= self.prob_th and
            min(region[2], region[3]) > 19]

        mark_img = np.array(orig_img, copy=True)
        for rect, prob, _ in candidates:
            x, y, w, h = rect
            cv2.rectangle(mark_img, (x, y), (x+w, y+h),
                          (255, round((float(prob)**5)*100), 0), 1, cv2.LINE_AA)

        rep_regions, rep_idxs = self.unify_similar_regions(candidates)


        for i, rect in enumerate(rep_regions):
            x, y, w, h = rect
            cv2.rectangle(mark_img, (x, y), (x+w, y+h),
                          (0, 255, 0), 2, cv2.LINE_AA)

        return rep_regions, mark_img, canny_img, imgt

    def unify_similar_regions(self, candidates):
        rep_regions = []
        rep_idxs = []
        skip = [False]*len(candidates)

        def pr_func(R): return self.region_priority(R)
        candidates = sorted(
            candidates, key=pr_func, reverse=True)

        for i in range(len(candidates)):
            if skip[i]:
                continue
            region1, _, idx1 = candidates[i]
            rep_reg = region1
            rep_idx = idx1
            for j in range(i+1, len(candidates)):
                if skip[j]:
                    continue
                region2, _, _= candidates[j]
                iou = self.iou(rep_reg, region2)

                if iou >= self.iou_th:
                    skip[j] = True
            rep_regions.append(rep_reg)
            rep_idxs.append(rep_idx)

        return rep_regions, rep_idxs

    def region_priority(self, R):
        #cdist_pr = 1.0/R[2]
        size_pr = (R[0][2]*R[0][3])**(1.5)
        prob_pr = R[1]**2
        squareness_pr = (min(R[0][2], R[0][3])/max(R[0][2], R[0][3]))**5
        return size_pr*prob_pr

    def extract_region(self, img, x, y, w, h):
        return img[y: y + h, x:x+w, ]

    def iou(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        if x1 <= x2 and y1 <= y2:
            return 1.0
        elif x1 >= x2 and y1 >= y2:
            return 1.0
        else:
            interx = max(0, (w1 if x1 <= x2 else w2) - abs(x1-x2))
            intery = max(0, (h1 if y1 <= y2 else h2) - abs(y1-y2))

        inter_area = interx*intery
        area1 = w1*h1
        area2 = w2*h2
        union_area = area1+area2 - inter_area
        return inter_area/union_area

    
    
    def rgb2hsv(self, rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


#%%
