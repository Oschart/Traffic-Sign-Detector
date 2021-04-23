
# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from math import sqrt

from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
import cv2
from utils import plot_showcase


class SignDetector:

    def __init__(self, pretrained=True):
        self.sign_classifier = SignClassifier(use_cached=pretrained)
        self.iou_th = 0.01
        self.prob_th = 0.2

    def detect_signs(self, img):
        ss = SelectiveSearch()
        regions = ss.propose_regions(img)

        num_chosen_regions = 500
        chosen_regions = regions

        region_slices = []
        region_probs = []
        for i, rect in enumerate(chosen_regions):
            x, y, w, h = rect
            region_slices.append(self.extract_region(img, x, y, w, h))

        region_probs = self.sign_classifier.predict(region_slices)

        self.prob_th = sorted(region_probs, reverse=True)[2]

        candidates = [(region, prob, i) for i, (region, prob) in enumerate(
            zip(regions, region_probs)) if prob >= self.prob_th and min(region[2],region[3]) > 15]

        print(candidates)

        mark_img = np.array(img, copy=True)
        for rect, prob, _ in candidates:
            x, y, w, h = rect
            print(round((float(prob)**5)*100))
            cv2.rectangle(mark_img, (x, y), (x+w, y+h),
                          (255, round((float(prob)**5)*100), 0), 1, cv2.LINE_AA)

        plt.imshow(mark_img)
        plt.show()

        rep_regions, rep_idxs = self.unify_similar_regions(candidates)

        for i in range(len(rep_regions)):
            for j in range(i+1, len(rep_regions)):
                print(
                    f'p1: {rep_regions[i]}, p2: {rep_regions[j]}, iou= {self.iou(rep_regions[i], rep_regions[j])}')

        for i, rect in enumerate(rep_regions):
            x, y, w, h = rect
            cv2.rectangle(mark_img, (x, y), (x+w, y+h),
                          (0, 255, 0), 1, cv2.LINE_AA)

        plt.imshow(mark_img)
        plt.show()
        return rep_regions

    def unify_similar_regions(self, candidates):
        rep_regions = []
        rep_idxs = []
        skip = [False]*len(candidates)

        pr_func = lambda p:  sqrt(p[0][2]*p[0][3])*(p[1]**5)*(min(p[0][2],p[0][3])/max(p[0][2],p[0][3]))**6
        candidates = sorted(
            candidates, key=pr_func, reverse=True)

        for i in range(len(candidates)):
            if skip[i]:
                continue
            region1, prob1, idx1 = candidates[i]
            area1 = region1[2]*region1[3]
            rep_reg = region1
            rep_idx = idx1
            for j in range(i+1, len(candidates)):
                if skip[j]:
                    continue
                region2, prob2, idx2 = candidates[j]
                iou = self.iou(rep_reg, region2)

                if iou >= self.iou_th:
                    skip[j] = True
            rep_regions.append(rep_reg)
            rep_idxs.append(rep_idx)

        return rep_regions, rep_idxs

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
