import cv2
import numpy as np
import os

def test_mask_grabcut_banana_very_wide():
    banana_frame = "pixel_frames/banana/frames/frame_0310.png"
    if os.path.exists(banana_frame):
        img = cv2.imread(banana_frame)
        h, w, _ = img.shape
        
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        mask = np.zeros((sh, sw), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        
        # Wide bounding box to prevent clipping
        x1, y1 = int(sw * 0.12), int(sh * 0.22)
        x2, y2 = int(sw * 0.88), int(sh * 0.68) 
        
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:sh, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:sw] = cv2.GC_BGD
        
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        # Yellow range
        lower_yellow = np.array([12, 60, 60])
        upper_yellow = np.array([38, 255, 255])
        yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
        
        # Also mark blue pixels as definitely background
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([135, 255, 255])
        blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
        mask[blue_pixels > 0] = cv2.GC_BGD
        
        # Also mark white highlights of the water bottle (which are on the right side) as background
        # Let's find white pixels near the right side of the bounding box
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        # Casing of the water bottle is bright white plastic
        mask[(gray > 180) & (mask != cv2.GC_BGD) & (np.arange(sw) > sw * 0.70)] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        final_mask = np.zeros_like(bin_mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask[labels == largest_label] = 255
        else:
            final_mask = bin_mask
            
        segmented = cv2.bitwise_and(img_small, img_small, mask=final_mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_mask_grabcut_banana_wide.png", segmented)
        print("Saved wide GrabCut banana to scratch/test_mask_grabcut_banana_wide.png")

if __name__ == "__main__":
    test_mask_grabcut_banana_very_wide()
