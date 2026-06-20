import cv2
import numpy as np
import os

def test_mask_grabcut_banana():
    banana_frame = "pixel_frames/banana/frames/frame_0310.png"
    if os.path.exists(banana_frame):
        img = cv2.imread(banana_frame)
        h, w, _ = img.shape
        
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        mask = np.zeros((sh, sw), np.uint8)
        mask[:] = cv2.GC_PR_BGD  # Default probably background
        
        # Bounding box around the central banana
        x1, y1 = int(sw * 0.15), int(sh * 0.25)
        x2, y2 = int(sw * 0.85), int(sh * 0.75)
        
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:sh, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:sw] = cv2.GC_BGD
        
        # Inside bounding box, use yellow HSV color to mark foreground
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 60, 60])
        upper_yellow = np.array([38, 255, 255])
        yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Mark yellow pixels as GC_PR_FGD or GC_FGD
        mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Clean up
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
        cv2.imwrite("scratch/test_mask_grabcut_banana.png", segmented)
        print("Saved mask GrabCut banana test to scratch/test_mask_grabcut_banana.png")

if __name__ == "__main__":
    test_mask_grabcut_banana()
