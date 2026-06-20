import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import get_headphone_mask

def render_all_masks():
    frames_dir = "pixel_frames/headphone/frames"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    
    output_dir = "scratch/headphone_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Segmenting all {len(frame_files)} headphone frames...")
    for f in frame_files:
        img_path = os.path.join(frames_dir, f)
        img = cv2.imread(img_path)
        mask = get_headphone_mask(img)
        
        # Calculate mask area and touch boundaries for diagnostics
        h, w = mask.shape
        area = np.sum(mask > 0)
        touches_left = np.any(mask[:, :int(w * 0.05)] > 0)
        touches_right = np.any(mask[:, -int(w * 0.05):] > 0)
        
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        # Draw bounding box and print diagnostics on the image
        cv2.putText(segmented, f"{f} - Area: {area}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 255), 3)
        cv2.putText(segmented, f"Touches L/R: {touches_left}/{touches_right}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 255), 2)
        
        out_path = os.path.join(output_dir, f"seg_{f}")
        cv2.imwrite(out_path, segmented)
        print(f"  Processed {f}: Area={area}, TouchesL/R={touches_left}/{touches_right}")

if __name__ == "__main__":
    render_all_masks()
