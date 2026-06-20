#!/usr/bin/env python3
"""
3D Video Generation Script (Improved)
Segments target objects, estimates smooth depth (Distance Transform + Sobel),
adds 3D extrusion thickness, and renders a slow-motion rotating 3D showcase video.
"""

import cv2
import os
import sys
import numpy as np
from PIL import Image

OUTPUT_BASE = "pixel_frames"

def get_foreground_mask(img, object_name):
    """
    Isolate target object from the background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if object_name == "banana":
        # HSV thresholding for yellow banana
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 70, 70])
        upper_yellow = np.array([38, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
    elif object_name == "headphone":
        # Adaptive dark pixel thresholding
        _, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Select largest central component
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            outer_mask = np.zeros_like(thresh)
            cv2.drawContours(outer_mask, [contours[0]], -1, 255, -1)
            
            # Keep raw dark pixels within the bounding shape (leaves holes open)
            mask = cv2.bitwise_and(thresh, outer_mask)
    else:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        mask = thresh
        
    return mask


def estimate_depth(img, mask):
    """
    Generate a smooth, rounded depth map using a blend of Distance Transform 
    and gradients inside the segmented mask.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Distance transform (smooth rounded bulge)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Gradients (surface details)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
    
    # Blend: 70% volume bulge, 30% surface details
    blended = cv2.addWeighted(dist_norm, 0.7, grad_norm, 0.3, 0)
    
    # Restrict to mask
    depth = cv2.bitwise_and(blended, blended, mask=mask)
    return depth


def render_3d_frame(img, depth, mask, angle=0.0, depth_scale=70, thickness=20, num_layers=4):
    """
    Project points to 3D with extrusion (thickness) layers, rotating around Y-axis.
    """
    h, w, _ = img.shape
    
    # Downsample for smooth rendering performance
    scale = 0.25
    small_w = int(w * scale)
    small_h = int(h * scale)
    
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    canvas_w, canvas_h = 1080, 1080
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]  # Dark background
    
    # Virtual camera scale & offset
    scale_x = 4.2
    scale_y = 3.2
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 100
    
    points = []
    
    for y in range(small_h):
        for x in range(small_w):
            if mask_small[y, x] > 0:
                d = (depth_small[y, x] / 255.0) * depth_scale
                
                # Center coordinates
                xc = (x - small_w // 2) * scale_x
                yc = (y - small_h // 2) * scale_y
                
                # Extrude backwards to create thickness (solid shell)
                for layer in range(num_layers):
                    # Z-offset goes backwards (negative Z direction)
                    z_offset = - (layer / max(1, num_layers - 1)) * thickness
                    zc = d + z_offset
                    
                    # Rotate Y-axis
                    rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                    rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                    rot_y = yc
                    
                    # Project to 2D canvas
                    px = int(rot_x + offset_x)
                    py = int(rot_y - rot_z * 0.4 + offset_y)
                    
                    # Darken back layers slightly to simulate ambient occlusion
                    shadow_factor = 1.0 - (layer / num_layers) * 0.4
                    b, g, r = img_small[y, x]
                    r_sh = int(r * shadow_factor)
                    g_sh = int(g * shadow_factor)
                    b_sh = int(b * shadow_factor)
                    
                    points.append((rot_z, px, py, r_sh, g_sh, b_sh))
                    
    # Painter's Algorithm: Sort back-to-front by projected rotation depth
    points.sort(key=lambda p: p[0])
    
    # Draw points
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size = max(2, int(3 + (z + depth_scale) / depth_scale * 1.5))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (b, g, r), -1)
            
    return canvas


def make_3d_video_showcase(object_name, total_frames=90, fps=15.0):
    """
    Select the best middle frame, extract it, and generate a high-quality,
    slow-motion 3D rotating showcase with thickness and zero jitter noise.
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
        
    # Take the middle frame as the most representative one
    best_frame_file = frame_files[len(frame_files) // 2]
    frame_path = os.path.join(frames_dir, best_frame_file)
    print(f"\nCreating 3D rotating showcase from single frame: {best_frame_file}")
    
    img = cv2.imread(frame_path)
    mask = get_foreground_mask(img, object_name)
    depth = estimate_depth(img, mask)
    
    video_output_path = os.path.join(obj_dir, f"{object_name}_3d_reconstruction.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (1080, 1080))
    
    # Render rotating frames (360 degrees rotation)
    for idx in range(total_frames):
        # Progressively rotate
        angle = (idx / total_frames) * 2 * np.pi
        
        # Render with extrusion thickness = 25
        rendered_frame = render_3d_frame(
            img, depth, mask, angle=angle, 
            depth_scale=80, thickness=25, num_layers=5
        )
        
        out.write(rendered_frame)
        if (idx + 1) % 15 == 0 or idx == 0:
            print(f"  Frame {idx+1}/{total_frames} rendered ({angle/np.pi*180:.1f}°)")
            
    out.release()
    print(f"\n🎉 3D Showcase Video saved successfully to:")
    print(f"   👉 {video_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_3d_video.py <object_name>")
        sys.exit(1)
        
    obj = sys.argv[1]
    # Default: 90 frames at 15fps = 6 seconds of slow, smooth rotation
    make_3d_video_showcase(obj, total_frames=90, fps=15.0)
