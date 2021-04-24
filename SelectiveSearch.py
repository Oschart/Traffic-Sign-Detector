import sys
import cv2


class SelectiveSearch:

    def propose_regions_countour(self, img):
        canny_img = cv2.Canny(img, 50, 150, apertureSize=3)
        contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        try: hierarchy = hierarchy[0]
        except: hierarchy = []

        regions = []
        # computes the bounding box for the contour, and draws it on the frame,
        for contour, hier in zip(contours, hierarchy):
            regions.append(cv2.boundingRect(contour))
        
        return regions, canny_img

    def propose_regions(self, img, fast_mode=True):
        # speed-up using multithreads
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(strategy_color)
        # set input image on which we will run segmentation
        ss.setBaseImage(img)

        # Switch to fast but low recall Selective Search method
        if fast_mode:
            ss.switchToSelectiveSearchFast()
        # Switch to high recall but slow Selective Search method
        else:
            ss.switchToSelectiveSearchQuality()

        # run selective search segmentation on input image
        rects = ss.process()
        print('Total Number of Region Proposals: {}'.format(len(rects)))

        return rects
        #cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
