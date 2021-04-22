import sys
import cv2

class SelectiveSearch:

    def propose_regions(self, img, fast_mode=True):
        # speed-up using multithreads
        cv2.setUseOptimized(True);
        cv2.setNumThreads(4);

        # create Selective Search Segmentation Object using default parameters
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
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

