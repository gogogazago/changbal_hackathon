import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import get_banana_mask, estimate_depth, render_3d_frame

def test_fusion_rotation():
    object_name = "banana"
    obj_dir = "pixel_frames/banana"
    frames_dir = os.path.join(obj_dir, "frames")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    if not frame_files:
        print("No frames found")
        return
        
    print(f"Loaded {len(frame_files)} frames for banana rotation test.")
    
    # We will generate a video of the same length as the number of frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("scratch/test_banana_fusion.mp4", fourcc, 10.0, (1080, 1080))
    
    # Estimate depth for all frames and render them with a constant camera angle (0.0)
    for idx, f in enumerate(frame_files):
        img_path = os.path.join(frames_dir, f)
        img = cv2.imread(img_path)
        mask = get_banana_mask(img)
        depth = estimate_depth(img, mask)
        
        # Render the 3D frame from a slightly tilted isometric view (constant angle = 0.0)
        # We can add a very small sway to make the 3D depth visible, or just keep it 0.0.
        rendered = render_3d_frame(
            img, depth, mask, angle=0.0, 
            depth_scale=85, thickness_factor=1.0, num_layers=5
        )
        
        out.write(rendered)
        print(f"  Rendered {idx+1}/{len(frame_files)}: {f}")
        
    out.release()
    print("Saved fusion video to scratch/test_banana_fusion.mp4")

if __name__ == "__main__":
    test_fusion_rotation()
