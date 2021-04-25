
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
        self.sign_classifier = SignClassifier(
            use_cached=pretrained, use_orig=True)
        self.iou_th = 0.1
        self.prob_th = 0.5
        self.casei = 1
        self.path = 'ss'

    def detect_signs(self, orig_img):

        imgtc = turnoff_irrelevant(orig_img)
        ss = SelectiveSearch()
        regions, canny_img = ss.propose_regions_countour(imgtc)
        #print(f'Number of proposed regions = {len(regions)}')

        region_slices = []
        region_probs = []
        for rect in regions:
            x, y, w, h = rect
            region_slices.append(self.extract_region(orig_img, x, y, w, h))

        region_probs = self.sign_classifier.predict(
            region_slices)

        #print(f'confidence threshold = {self.prob_th}')

        candidates = [(region, prob) for region, prob in 
            zip(regions, region_probs) if prob > self.prob_th and
            min(region[2], region[3]) >= 15]
        
        mark_img = np.array(orig_img, copy=True)

        rep_regions, rep_probs = self.unify_similar_regions(candidates)
        for rect, prob in zip(rep_regions, rep_probs):
            x, y, w, h = rect
            cv2.rectangle(mark_img, (x, y), (x+w, y+h),
                          (0, 255, 0), 2, cv2.LINE_AA)

        return rep_regions, mark_img, canny_img, imgtc

    def unify_similar_regions(self, candidates):
        rep_regions = []
        rep_probs = []
        skip = [False]*len(candidates)

        def pr_func(R): return self.region_priority(R)
        candidates = sorted(
            candidates, key=pr_func, reverse=True)

        for i in range(len(candidates)):
            if skip[i]:
                continue
            region1, prob1 = candidates[i]
            rep_reg = region1
            for j in range(i+1, len(candidates)):
                if skip[j]:
                    continue
                region2, _ = candidates[j]
                iou = self.iou(rep_reg, region2)

                if iou >= self.iou_th:
                    skip[j] = True
            rep_regions.append(rep_reg)
            rep_probs.append(prob1)

        return rep_regions, rep_probs

    def region_priority(self, R):
        size_pr = (R[0][2]*R[0][3])
        prob_pr = R[1]
        squareness_pr = min(R[0][2], R[0][3])/max(R[0][2], R[0][3])
        return size_pr*prob_pr*squareness_pr

    def extract_region(self, img, x, y, w, h):
        return img[y: y + h, x:x+w, ]

    def iou(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        x_left = min(x1, x2)
        x_right = max(x1+w1, x2+w2)
        y_top = min(y1, y2)
        y_bottom = max(y1+h1, y2+h2)

        area1 = w1*h1
        area2 = w2*h2
        inter_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = area1+area2 - inter_area

        return inter_area/union_area

# %%
