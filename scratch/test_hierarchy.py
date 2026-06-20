import cv2
import numpy as np
import os

def test_contour_hole_filling():
    hp_frame = "pixel_frames/headphone/frames/frame_0140.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        h, w, _ = img.shape
        
        # Downsample for processing
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        # GrabCut bounding box
        rect = (int(sw * 0.08), int(sh * 0.15), int(sw * 0.84), int(sh * 0.7))
        mask_gc = np.zeros((sh, sw), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img_small, mask_gc, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        gc_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
        
        # Smooth the GrabCut mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gc_mask_closed = cv2.morphologyEx(gc_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours with hierarchy
        # RETR_CCOMP retrieves all contours and organizes them into a two-level hierarchy:
        # - Level 1: Outer boundaries
        # - Level 2: Boundaries of holes
        contours, hierarchy = cv2.findContours(gc_mask_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = gc_mask_closed.copy()
        
        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for idx, contour in enumerate(contours):
                # Check if it is a hole (has a parent contour)
                parent_idx = hierarchy[idx][3]
                if parent_idx != -1:
                    # It's a hole!
                    area = cv2.contourArea(contour)
                    # If the hole is large enough (like the inside of the headband), mask it out
                    if area > 1000:
                        print(f"Hole found with area: {area}, masking it out.")
                        cv2.drawContours(mask, [contour], -1, 0, -1)
                        
        segmented = cv2.bitwise_and(img_small, img_small, mask=mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_headphone_hierarchy.png", segmented)
        print("Saved hierarchy segmented image to scratch/test_headphone_hierarchy.png")

if __name__ == "__main__":
    test_contour_hole_filling()
