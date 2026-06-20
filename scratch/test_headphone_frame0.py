import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import get_headphone_mask

def test_headphone_frame0():
    img_path = "pixel_frames/headphone/frames/frame_0000.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        mask = get_headphone_mask(img)
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_headphone_frame0_seg.png", segmented)
        print("Saved segmented headphone frame 0 to scratch/test_headphone_frame0_seg.png")

if __name__ == "__main__":
    test_headphone_frame0()
