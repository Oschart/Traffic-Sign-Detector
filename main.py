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

def run(tests_path='test_cases'):
    sd = SignDetector(pretrained=True)

    tests = [test.name for test in os.scandir(
                tests_path) if test.is_file()]
    for i, img_fname in tqdm(enumerate(tests)):
        tdir = '/'.join([tests_path, img_fname])
        save_dir = '/'.join(['test_results', img_fname])
        save_dirc = '/'.join(['test_results_canny', img_fname])
        save_dirt = '/'.join(['test_results_t', img_fname])
        print(f'Test #{i+1} ===>> {img_fname}')
        img = cv2.imread(tdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes, annotated_img, canny_img, imgt = sd.detect_signs(img)
        plt.imsave(save_dir, annotated_img)
        plt.imsave(save_dirc, canny_img)
        plt.imsave(save_dirt, imgt)
        print('Bounding boxes (x,y,w,h):')
        print(bboxes)


run()