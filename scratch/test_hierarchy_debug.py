import cv2
import numpy as np
import os

def test_contour_debug():
    hp_frame = "pixel_frames/headphone/frames/frame_0140.png"
    img = cv2.imread(hp_frame)
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    rect = (int(sw * 0.08), int(sh * 0.15), int(sw * 0.84), int(sh * 0.7))
    mask_gc = np.zeros((sh, sw), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_small, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    gc_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gc_mask_closed = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite("scratch/debug_mask_before.png", gc_mask_closed)
    
    contours, hierarchy = cv2.findContours(gc_mask_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = gc_mask_closed.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for idx, contour in enumerate(contours):
            parent_idx = hierarchy[idx][3]
            if parent_idx != -1:
                area = cv2.contourArea(contour)
                if area > 1000:
                    print(f"Hole found with area: {area}")
                    # Draw contour with color 0 (black), thickness -1 (filled)
                    cv2.drawContours(mask, contours, idx, 0, -1)
                    
    cv2.imwrite("scratch/debug_mask_after.png", mask)
    segmented = cv2.bitwise_and(img_small, img_small, mask=mask)
    cv2.imwrite("scratch/debug_segmented.png", segmented)
    print("Saved debug masks to scratch/")

if __name__ == "__main__":
    test_contour_debug()
