#!/usr/bin/env python3
"""
3D Video Generation Script
Segments the target object from video frames, projects it into 3D using depth maps,
removes the background (table, chairs), and renders it as a floating 3D video.
"""

import cv2
import os
import sys
import numpy as np
from PIL import Image

OUTPUT_BASE = "pixel_frames"

def get_foreground_mask(img, object_name):
    """
    Generate a binary mask isolating the target object from the background.
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if object_name == "banana":
        # Segment yellow color in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 70, 70])
        upper_yellow = np.array([38, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    elif object_name == "headphone":
        # Segment dark headphones on light desk
        _, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Keep only the component that overlaps with the largest central contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            # Largest contour is the headphones
            largest_contour = contours[0]
            
            # Create a solid mask for the bounding shape
            outer_mask = np.zeros_like(thresh)
            cv2.drawContours(outer_mask, [largest_contour], -1, 255, -1)
            
            # Intersect raw threshold (preserves internal holes/desk gaps) with outer mask
            mask = cv2.bitwise_and(thresh, outer_mask)
    else:
        # Default fallback: Center region thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        mask = thresh
        
    return mask


def estimate_depth(img, mask):
    """
    Estimate relative depth map focusing on the masked object.
    Uses luminance gradient + center weighting inside the mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Compute Sobel gradients as depth/surface cues
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Blur depth map for smooth transitions
    depth_base = cv2.GaussianBlur(magnitude, (15, 15), 0)
    
    # Apply center weighting (assume center of the mask is closer)
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    # Find center of mass of the mask
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
        
    dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_dist = dist_from_center.max() if dist_from_center.max() > 0 else 1.0
    center_weight = 1.0 - (dist_from_center / max_dist)
    
    # Combine gradient depth and center weighting
    depth = (depth_base / 255.0 * 0.5 + center_weight * 0.5)
    
    # Normalize depth map
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Only keep depth inside mask
    depth = cv2.bitwise_and(depth, depth, mask=mask)
    return depth


def render_3d_frame(img, depth, mask, angle=0.0, depth_scale=60):
    """
    Project the masked foreground pixels into 3D space using the depth map,
    and render a frame from a specific camera angle.
    """
    h, w, _ = img.shape
    
    # Downsample for rendering speed
    scale = 0.25
    small_w = int(w * scale)
    small_h = int(h * scale)
    
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # Render canvas (Dark background)
    canvas_w, canvas_h = 1080, 1080
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]  # Charcoal background
    
    # Camera projection params
    scale_x = 4.0
    scale_y = 3.0
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 100
    
    # Collect 3D points
    points = []
    for y in range(small_h):
        for x in range(small_w):
            if mask_small[y, x] > 0:
                d = (depth_small[y, x] / 255.0) * depth_scale
                
                # Center coordinates
                xc = (x - small_w // 2) * scale_x
                yc = (y - small_h // 2) * scale_y
                zc = d
                
                # Rotate around Y axis
                rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                rot_y = yc
                
                # Isometric/Perspective projection on 2D screen
                px = int(rot_x + offset_x)
                py = int(rot_y - rot_z * 0.5 + offset_y)
                
                # Colors
                b, g, r = img_small[y, x]
                points.append((rot_z, px, py, r, g, b))
                
    # Sort points back-to-front (painter's algorithm)
    points.sort(key=lambda p: p[0])
    
    # Draw points
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            # Draw point block size based on depth
            size = max(2, int(3 + (z + depth_scale) / depth_scale))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (int(b), int(g), int(r)), -1)
            
    # Add object label
    cv2.putText(canvas, f"3D Volumetric: {img.shape[1]}x{img.shape[0]} source", (50, canvas_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 180), 1, cv2.LINE_AA)
                
    return canvas


def make_3d_video(object_name):
    """
    Process all extracted frames for an object, segment them,
    and create a 3D rotating rendering video.
    """
    obj_dir = os.path.join(OUTPUT_BASE, object_name)
    frames_dir = os.path.join(obj_dir, "frames")
    
    if not os.path.exists(frames_dir):
        print(f"❌ Error: Frames directory not found: {frames_dir}")
        return
        
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    if not frame_files:
        print(f"❌ Error: No frames found in {frames_dir}")
        return
        
    print(f"\nProcessing {len(frame_files)} frames for '{object_name}'...")
    
    # Output video file
    video_output_path = os.path.join(obj_dir, f"{object_name}_3d_reconstruction.mp4")
    
    # Create video writer (1080x1080 resolution, 15 FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 15.0, (1080, 1080))
    
    # Process and write frames
    for idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        
        # Step 1: Segment foreground
        mask = get_foreground_mask(img, object_name)
        
        # Step 2: Estimate depth focusing on segmented object
        depth = estimate_depth(img, mask)
        
        # Step 3: Render 3D point cloud frame
        # Camera rotates slowly over time
        angle = (idx / len(frame_files)) * 2 * np.pi
        rendered_frame = render_3d_frame(img, depth, mask, angle=angle)
        
        out.write(rendered_frame)
        print(f"  Frame {idx+1}/{len(frame_files)} rendered (angle: {angle:.2f} rad)")
        
    out.release()
    print(f"\n🎉 3D Reconstruction video saved successfully to:")
    print(f"   👉 {video_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_3d_video.py <object_name>")
        print("Example: python3 create_3d_video.py headphone")
        sys.exit(1)
        
    obj = sys.argv[1]
    make_3d_video(obj)
