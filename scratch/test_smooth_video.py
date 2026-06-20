import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import get_headphone_mask, estimate_depth, render_3d_frame

def test_smooth_headphone():
    video_path = "video/headphone/video_headphone.mp4"
    if not os.path.exists(video_path):
        print("Source video not found")
        return
        
    cap = cv2.VideoCapture(video_path)
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Source video has {total_source_frames} frames, FPS: {fps}")
    
    # We want to output a smooth 30 fps video. 
    # Let's sample every 2nd frame to get ~137 frames.
    step = 2
    output_fps = 20.0 # Slow it down slightly for cinematic effect
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs("scratch", exist_ok=True)
    out = cv2.VideoWriter("scratch/test_headphone_smooth.mp4", fourcc, output_fps, (1080, 1080))
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            # Segment and estimate depth
            mask = get_headphone_mask(frame)
            depth = estimate_depth(frame, mask)
            
            # Smooth the depth map using a Gaussian blur to eliminate high-frequency spikes
            depth_smoothed = cv2.GaussianBlur(depth, (15, 15), 0)
            
            # Render the 3D frame with slightly larger splat size to close gaps
            # We add a small sway animation
            sway_angle = 0.08 * np.sin(saved_count * 0.1)
            
            # Render using custom rendering with larger splats
            rendered = render_3d_frame(
                frame, depth_smoothed, mask, angle=sway_angle, 
                depth_scale=85, thickness_factor=0.8, num_layers=5
            )
            
            out.write(rendered)
            saved_count += 1
            print(f"  Processed frame {frame_count}/{total_source_frames} -> output {saved_count}")
            
            # Limit to 60 frames for quick testing
            if saved_count >= 60:
                break
                
        frame_count += 1
        
    cap.release()
    out.release()
    print("Saved smooth test video to scratch/test_headphone_smooth.mp4")

if __name__ == "__main__":
    test_smooth_headphone()
