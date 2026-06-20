import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import download_midas_if_needed, get_headphone_mask, get_banana_mask

def analyze_depth_thresholds(object_name, frame_name):
    img_path = f"pixel_frames/{object_name}/frames/{frame_name}"
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return
        
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Run raw MiDaS
    model_path = download_midas_if_needed()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    raw_depth = net.forward()[0, :, :]
    
    # Resize to image shape
    raw_depth_resized = cv2.resize(raw_depth, (w, h))
    
    # Normalize globally for printing/inspecting
    depth_norm = cv2.normalize(raw_depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Get current masks
    if object_name == "headphone":
        mask = get_headphone_mask(img)
    else:
        mask = get_banana_mask(img)
        
    fg_depths = depth_norm[mask == 255]
    bg_depths = depth_norm[mask == 0]
    
    print(f"\n=== Depth analysis for '{object_name}' ({frame_name}) ===")
    if len(fg_depths) > 0:
        print(f"Foreground Object Depth: Min={np.min(fg_depths)}, Max={np.max(fg_depths)}, Mean={np.mean(fg_depths):.1f}")
    if len(bg_depths) > 0:
        print(f"Background Table/Desk Depth: Min={np.min(bg_depths)}, Max={np.max(bg_depths)}, Mean={np.mean(bg_depths):.1f}")
        
    # Let's save a visualization showing depth thresholds: depth > 100, depth > 120, etc.
    os.makedirs("scratch", exist_ok=True)
    for t in [90, 110, 130]:
        mask_t = np.where(depth_norm > t, 255, 0).astype(np.uint8)
        segmented_t = cv2.bitwise_and(img, img, mask=mask_t)
        cv2.imwrite(f"scratch/depth_thresh_{object_name}_{t}.png", segmented_t)
        print(f"Saved scratch/depth_thresh_{object_name}_{t}.png")

if __name__ == "__main__":
    analyze_depth_thresholds("headphone", "frame_0100.png")
    analyze_depth_thresholds("banana", "frame_0100.png")
