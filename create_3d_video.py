#!/usr/bin/env python3
"""
3D Video Generation Script (Perfect Boundary Reconstruction)
Segments target objects (headphones, banana) with absolute detail preservation,
estimates smooth volumetric depth, and renders a slow-motion rotating 3D video.
"""

import cv2
import os
import sys
import numpy as np
from PIL import Image

OUTPUT_BASE = "pixel_frames"

def get_headphone_mask(img):
    """
    Isolate headphones using mask GrabCut. Preserves metallic highlights 
    and removes the desk inside the loop headband.
    """
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD  # Default probably background
    
    # Bounding box around headphones
    x1, y1 = int(sw * 0.08), int(sh * 0.15)
    x2, y2 = int(sw * 0.92), int(sh * 0.85)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    mask[(gray < 100) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
    mask[(gray < 65) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    except:
        bin_mask = np.where(gray < 100, 255, 0).astype('uint8')
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    final_mask = np.zeros_like(bin_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask[labels == largest_label] = 255
    else:
        final_mask = bin_mask
        
    return cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def get_banana_mask(img):
    """
    Isolate banana using mask GrabCut. Crops out water bottle and ignores blue colors.
    """
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    # Bounding box cropped on the right to exclude water bottle
    x1, y1 = int(sw * 0.12), int(sh * 0.22)
    x2, y2 = int(sw * 0.88), int(sh * 0.68)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([12, 60, 60])
    upper_yellow = np.array([38, 255, 255])
    yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
    mask[blue_pixels > 0] = cv2.GC_BGD
    
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    mask[(gray > 180) & (mask != cv2.GC_BGD) & (np.arange(sw) > sw * 0.70)] = cv2.GC_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    except:
        bin_mask = yellow_pixels
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    final_mask = np.zeros_like(bin_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask[labels == largest_label] = 255
    else:
        final_mask = bin_mask
        
    return cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def estimate_depth(img, mask):
    """
    Generate smooth volumetric depth (bulge) blended with Sobel details.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Distance transform for circular 3D volume
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Details
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
    
    # Blend: 80% smooth shape, 20% detail
    blended = cv2.addWeighted(dist_norm, 0.8, grad_norm, 0.2, 0)
    return cv2.bitwise_and(blended, blended, mask=mask)


def render_3d_frame(img, depth, mask, angle=0.0, depth_scale=80, thickness_factor=0.6, num_layers=5):
    """
    Project 3D points with round/cylindrical volume thickness, Y-axis rotation.
    """
    h, w, _ = img.shape
    scale = 0.25
    small_w = int(w * scale)
    small_h = int(h * scale)
    
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    dist_map = cv2.distanceTransform(mask_small, cv2.DIST_L2, 5)
    dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
    
    canvas_w, canvas_h = 1080, 1080
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]  # Background
    
    scale_x = 4.2
    scale_y = 3.2
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 100
    
    points = []
    
    for y in range(small_h):
        for x in range(small_w):
            if mask_small[y, x] > 0:
                d = (depth_small[y, x] / 255.0) * depth_scale
                
                # Hemispherical thickness profile based on boundary distance
                max_t = dist_map[y, x] * depth_scale * thickness_factor
                
                xc = (x - small_w // 2) * scale_x
                yc = (y - small_h // 2) * scale_y
                
                for layer in range(num_layers):
                    # Z-extrusion curves backwards
                    t_offset = (layer / max(1, num_layers - 1)) * max_t
                    zc = d - t_offset
                    
                    # Orbit rotation
                    rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                    rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                    rot_y = yc
                    
                    px = int(rot_x + offset_x)
                    py = int(rot_y - rot_z * 0.4 + offset_y)
                    
                    # Ambient shadow occlusion
                    shadow = 1.0 - (layer / num_layers) * 0.45
                    b, g, r = img_small[y, x]
                    r_sh = int(r * shadow)
                    g_sh = int(g * shadow)
                    b_sh = int(b * shadow)
                    
                    points.append((rot_z, px, py, r_sh, g_sh, b_sh))
                    
    # Painter's Algorithm
    points.sort(key=lambda p: p[0])
    
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size = max(2, int(3 + (z + depth_scale) / depth_scale * 1.5))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (b, g, r), -1)
            
    return canvas


def make_3d_video_showcase(object_name, total_frames=90, fps=15.0):
    """
    Picks the absolute best frame to avoid occlusions, segments cleanly, 
    and generates the 3D rotating showcase video.
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
        
    # Selection logic based on object:
    if object_name == "banana":
        # First frame (frame_0000.png) has the banana fully in front of the bottle (unoccluded)
        best_frame_file = frame_files[0]
        thickness_val = 1.0 # Bananas are round tubes
    elif object_name == "headphone":
        # Middle frame (frame_0140.png) has the most complete symmetric view of the headphone
        best_frame_file = frame_files[len(frame_files) // 2]
        thickness_val = 0.8
    else:
        best_frame_file = frame_files[len(frame_files) // 2]
        thickness_val = 0.6
        
    frame_path = os.path.join(frames_dir, best_frame_file)
    print(f"\nCreating 3D Rotating Video for '{object_name}' using: {best_frame_file}")
    
    img = cv2.imread(frame_path)
    
    # Run the appropriate detailed mask generator
    if object_name == "headphone":
        mask = get_headphone_mask(img)
    elif object_name == "banana":
        mask = get_banana_mask(img)
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        
    depth = estimate_depth(img, mask)
    
    video_output_path = os.path.join(obj_dir, f"{object_name}_3d_reconstruction.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (1080, 1080))
    
    for idx in range(total_frames):
        angle = (idx / total_frames) * 2 * np.pi
        
        rendered_frame = render_3d_frame(
            img, depth, mask, angle=angle, 
            depth_scale=85, thickness_factor=thickness_val, num_layers=5
        )
        
        out.write(rendered_frame)
        if (idx + 1) % 15 == 0 or idx == 0:
            print(f"  Frame {idx+1}/{total_frames} rendered ({angle/np.pi*180:.1f}°)")
            
    out.release()
    print(f"\n🎉 3D Video saved successfully to:")
    print(f"   👉 {video_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_3d_video.py <object_name>")
        sys.exit(1)
        
    obj = sys.argv[1]
    make_3d_video_showcase(obj, total_frames=90, fps=15.0)
