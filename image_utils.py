import cv2
import numpy as np
import matplotlib.pyplot as plt


def turnoff_irrelevant(img, no_change=False):
    if no_change:
        return img
    
    cimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rel_colors = [
        [(0,50,20), (12,255,255)],
        [(175,50,20), (180,255,255)],
    ]
    mask = 0
    for [low, hi] in rel_colors:
        mask += cv2.inRange(cimg, low, hi)
    
    out_img = img.copy()
    out_img[np.where(mask==0)] = 0
    
    return smooth(out_img)

def smooth(img):
    return cv2.GaussianBlur(img,(5,5),0)

