import cv2
import numpy as np
import os

def test_segmentation():
    # Test headphone
    hp_frame = "pixel_frames/headphone/frames/frame_0000.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Headphone is dark on light background.
        # Let's try thresholding.
        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour in the center area
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        if contours:
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Take the largest one
            cv2.drawContours(mask, [contours[0]], -1, 255, -1)
            
        segmented = cv2.bitwise_and(img, img, mask=mask)
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_headphone_seg.png", segmented)
        print("Saved headphone segmentation test to scratch/test_headphone_seg.png")

    # Test banana
    banana_frame = "pixel_frames/banana/frames/frame_0000.png"
    if os.path.exists(banana_frame):
        img = cv2.imread(banana_frame)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Yellow color range in HSV
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        segmented = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite("scratch/test_banana_seg.png", segmented)
        print("Saved banana segmentation test to scratch/test_banana_seg.png")

if __name__ == "__main__":
    test_segmentation()
