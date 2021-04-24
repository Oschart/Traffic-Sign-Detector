import cv2
import numpy as np


def turnoff_irrelevant(img):

        cimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        rel_colors = [
            [(0,50,20), (30,255,255)],
            [(175,50,20), (180,255,255)],
            [(10, 100, 50), (30, 255, 255)],
        ]
        mask = 0
        for [low, hi] in rel_colors:
            mask += cv2.inRange(cimg, low, hi)
        
        out_img = img.copy()
        out_img[np.where(mask==0)] = 0
        return sharpen(out_img)

def sharpen(img):
    shrk = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.GaussianBlur(img,(5,5),0)