import cv2
import numpy as np
import os

def test_segmentation_improved():
    hp_frame = "pixel_frames/headphone/frames/frame_0000.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get dark pixels
        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Instead of filling the contour, let's keep only the components 
        # that overlap with the largest contour to remove background noise,
        # but keep the holes!
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Create a mask of the largest bounding region
            outer_mask = np.zeros_like(thresh)
            cv2.drawContours(outer_mask, [contours[0]], -1, 255, -1)
            
            # Intersect raw threshold with the outer mask
            final_mask = cv2.bitwise_and(thresh, outer_mask)
        else:
            final_mask = thresh
            
        segmented = cv2.bitwise_and(img, img, mask=final_mask)
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_headphone_seg_clean.png", segmented)
        print("Saved cleaned headphone segmentation to scratch/test_headphone_seg_clean.png")

if __name__ == "__main__":
    test_segmentation_improved()
