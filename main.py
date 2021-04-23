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




img_path = 'test_cases/Am_Rojo0024.jpg'
orig_img = cv2.imread(img_path)


newHeight = 800
newWidth = int(orig_img.shape[1]*newHeight/orig_img.shape[0])
orig_img = cv2.resize(orig_img, (newWidth, newHeight))
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

sd = SignDetector(pretrained=True)
print(sd.detect_signs(orig_img))
