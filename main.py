# %%
import matplotlib.pyplot as plt
import numpy as np
#import cv2
from skimage.io import imread, imshow


from SelectiveSearch import SelectiveSearch
from SignClassifier import SignClassifier
from SignDetector import SignDetector
import cv2
from tqdm import tqdm
import os
from plot_utils import plot_showcase

def run(tests_path='test_cases', pretrained=True, view_sample=[1, 2, 14, 16, 21], verbose=False):
    sd = SignDetector(pretrained=pretrained)

    tests = [test.name for test in os.scandir(
                tests_path) if test.is_file()]
    sid = 1
    for i, img_fname in enumerate(tests):
        tdir = '/'.join([tests_path, img_fname])
        save_dir = '/'.join(['test_results', img_fname])
        if verbose:
            print(f'Test #{i+1} ===>> {img_fname}')
        
        img = cv2.imread(tdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bboxes, annotated_img, canny_img, imgt = sd.detect_signs(img)
        plt.imsave(save_dir, annotated_img)
        if verbose:
            print('Bounding boxes (x,y,w,h):')
            print(bboxes)
        if i in view_sample:
            plot_showcase([img, imgt, canny_img, annotated_img], sid=sid, gray_list=[2])
            sid += 1
