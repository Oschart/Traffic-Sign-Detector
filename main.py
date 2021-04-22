# %%
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from skimage.io import imread, imshow


from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
from SignDetector import SignDetector
import cv2
from utils import plot_showcase





sc = SignClassifier(use_cached=True)


img_path = 'test_cases/Am_Rojo0002.jpg'
orig_img = cv2.imread(img_path)
orig_img = cv2.resize(orig_img, (500, 500))

sd = SignDetector()
sd.detect_signs(orig_img)