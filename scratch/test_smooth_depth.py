import cv2
import numpy as np
import os

def test_smooth_depth():
    hp_frame = "pixel_frames/headphone/frames/frame_0000.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Segment headphone
        _, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            outer_mask = np.zeros_like(thresh)
            cv2.drawContours(outer_mask, [contours[0]], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, outer_mask)
            
        # 1. Distance transform (smooth bulgy depth)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. Gradient-based depth (details)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
        
        # Blend: 70% smooth bulge, 30% details
        blended_depth = cv2.addWeighted(dist_norm, 0.7, grad_norm, 0.3, 0)
        blended_depth = cv2.bitwise_and(blended_depth, blended_depth, mask=mask)
        
        # Apply colormap to see it
        depth_colored = cv2.applyColorMap(blended_depth, cv2.COLORMAP_INFERNO)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_smooth_depth.png", depth_colored)
        print("Saved smooth depth test to scratch/test_smooth_depth.png")

if __name__ == "__main__":
    test_smooth_depth()
