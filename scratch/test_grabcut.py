import cv2
import numpy as np
import os

def test_grabcut_headphone():
    hp_frame = "pixel_frames/headphone/frames/frame_0140.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        h, w, _ = img.shape
        
        # Downsample for speed
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        # Define GrabCut bounding box around the center (where headphones are)
        # Bounding box coordinates: [x, y, w, h]
        # Headphones are in the middle 70% of the image
        rect = (int(sw * 0.1), int(sh * 0.2), int(sw * 0.8), int(sh * 0.6))
        
        mask = np.zeros((sh, sw), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Run GrabCut for 5 iterations
        cv2.grabCut(img_small, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Mask where background/prob_background is 0, foreground/prob_foreground is 1
        # 0 = GC_BGD, 1 = GC_FGD, 2 = GC_PR_BGD, 3 = GC_PR_FGD
        bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        
        segmented = cv2.bitwise_and(img_small, img_small, mask=bin_mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_grabcut_headphone.png", segmented)
        print("Saved GrabCut headphone segmentation to scratch/test_grabcut_headphone.png")

if __name__ == "__main__":
    test_grabcut_headphone()
