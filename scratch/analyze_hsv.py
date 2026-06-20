import cv2
import numpy as np
import os

def analyze_hsv():
    img_path = "pixel_frames/banana/frames/frame_0000.png"
    img = cv2.imread(img_path)
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # Let's save the Saturation channel as a grayscale image to visualize
    s_channel = hsv[:, :, 1]
    cv2.imwrite("scratch/banana_saturation.png", s_channel)
    print("Saved Saturation channel visualization to scratch/banana_saturation.png")
    
    # Also Value channel
    v_channel = hsv[:, :, 2]
    cv2.imwrite("scratch/banana_value.png", v_channel)
    print("Saved Value channel visualization to scratch/banana_value.png")

if __name__ == "__main__":
    analyze_hsv()
