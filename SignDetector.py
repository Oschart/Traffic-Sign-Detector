
# %%
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow


from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
import cv2
from utils import plot_showcase

class SignDetector:

    def __init__(self, pretrained = True):
        self.sign_classifier = SignClassifier(use_cached=pretrained)
        self.iou_th = 0.2
        self.prob_th = 0.9

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
        rep_regions = self.unify_similar_regions(chosen_regions, region_probs)

        mark_img = np.array(img, copy=True)

        for i, rect in enumerate(rep_regions):
            x, y, w, h = rect
            cv2.rectangle(mark_img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

        plt.imshow(mark_img)
        plt.show()


    def unify_similar_regions(self, regions, region_probs):
        rep_regions = []
        is_grouped = [False]*len(regions)
        for i in range(len(regions)):
            if is_grouped[i]:
                continue
            mx_prob = region_probs[i]
            rep_reg = regions[i]
            for j in range(i+1, len(regions)):
                if is_grouped[j]:
                    continue
                iou = self.iou(regions[i], regions[j])
                if iou >= self.iou_th:
                    if region_probs[j] > mx_prob:
                        mx_prob = region_probs[j]
                        rep_reg = regions[j]
                    is_grouped[j] = True
            if mx_prob >= self.prob_th:
                rep_regions.append(rep_reg)
        return rep_regions



    def extract_region(self, img, x, y, w, h):
        return img[y: y + h, x:x+w, ]

    def iou(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        interx = max(0, (w1 if x1 <= x2 else w2) - abs(x1-x2))
        intery = max(0, (h1 if y1 <= y2 else h2) - abs(y1-y2))

        inter_area = interx*intery
        area1 = w1*h1
        area2 = w2*h2
        union_area = area1+area2 - inter_area
        return inter_area/union_area