#!/usr/bin/env python3
"""
3D Video Generation Script (Robust Segmentation & Volume Extrusion)
Addresses:
1. Detailed segmentation (preserving hinges and edges) using GrabCut + Threshold subtraction.
2. Smooth, rounded 3D thickness (avoiding flat paper cutout look).
3. Slow-motion, zero-jitter showcase video.
"""

import cv2
import os
import sys
import numpy as np
from PIL import Image

OUTPUT_BASE = "pixel_frames"

def get_foreground_mask_detailed(img, object_name):
    """
    Generate a detailed binary mask.
    - Headphones: GrabCut (preserves details/hinges) + Grayscale threshold subtraction (removes desk inside loop).
    - Banana: HSV (isolates yellow) + Largest connected component.
    """
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if object_name == "headphone":
        # 1. GrabCut for detailed outer contour (including metallic hinges)
        rect = (int(w * 0.08), int(h * 0.15), int(w * 0.84), int(h * 0.7))
        mask_gc = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
            gc_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
        except Exception:
            # Fallback if GrabCut fails
            gc_mask = np.ones((h, w), np.uint8) * 255
            
        # 2. Threshold mask to identify the white/light desk surface
        # Desk is bright, headphones are dark
        _, desk_mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        
        # 3. Detailed mask = GC outer shape MINUS the desk surface
        # This cleans up the desk area inside the headphone loop
        mask = cv2.bitwise_and(gc_mask, cv2.bitwise_not(desk_mask))
        
        # Keep only the largest connected component to remove background noise
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        mask_clean = np.zeros_like(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean[labels == largest_label] = 255
        else:
            mask_clean = mask
            
        # Morphological closing to fill small gaps in the headband/hinge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
    elif object_name == "banana":
        # HSV thresholding for yellow color
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 70, 70])
        upper_yellow = np.array([38, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only the largest component (removes water bottle label or background)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        mask_clean = np.zeros_like(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean[labels == largest_label] = 255
            mask = mask_clean
            
    else:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        mask = thresh
        
    return mask


def estimate_depth(img, mask):
    """
    Generate a smooth, rounded depth map using Distance Transform 
    blended with Sobel gradients.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Distance transform (smooth rounded volume bulge)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 2. Gradients (surface detail)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
    
    # Blend: 80% volume bulge, 20% surface detail
    blended = cv2.addWeighted(dist_norm, 0.8, grad_norm, 0.2, 0)
    return cv2.bitwise_and(blended, blended, mask=mask)


def render_3d_frame(img, depth, mask, angle=0.0, depth_scale=80, thickness_factor=0.6, num_layers=5):
    """
    Project points to 3D with curved volume extrusion (thickness), Y-axis rotation.
    - Curved extrusion: Backwards extrusion is scaled by Distance Transform
      to make the back look rounded (like a tube/cylinder) instead of a flat extrusion.
    """
    h, w, _ = img.shape
    
    # Downsample for rendering speed
    scale = 0.25
    small_w = int(w * scale)
    small_h = int(h * scale)
    
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # Calculate distance transform on small mask for curved thickness
    dist_map = cv2.distanceTransform(mask_small, cv2.DIST_L2, 5)
    dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
    
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
                
                # Maximum thickness at this pixel is based on its distance to border
                # (Creates a circular cross-section / tube effect)
                max_t = dist_map[y, x] * depth_scale * thickness_factor
                
                # Center coordinates
                xc = (x - small_w // 2) * scale_x
                yc = (y - small_h // 2) * scale_y
                
                # Render multiple curved layers backwards
                for layer in range(num_layers):
                    # Extrusion profile: circular bulge backwards
                    # zc goes from front surface (d) to back surface (d - max_t)
                    t_offset = (layer / max(1, num_layers - 1)) * max_t
                    zc = d - t_offset
                    
                    # Y-axis rotation
                    rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                    rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                    rot_y = yc
                    
                    # Project to 2D
                    px = int(rot_x + offset_x)
                    py = int(rot_y - rot_z * 0.4 + offset_y)
                    
                    # Ambient occlusion: back layers are darker
                    shadow = 1.0 - (layer / num_layers) * 0.45
                    b, g, r = img_small[y, x]
                    r_sh = int(r * shadow)
                    g_sh = int(g * shadow)
                    b_sh = int(b * shadow)
                    
                    points.append((rot_z, px, py, r_sh, g_sh, b_sh))
                    
    # Painter's Algorithm: Sort back-to-front by projected Z depth
    points.sort(key=lambda p: p[0])
    
    # Draw points
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size = max(2, int(3 + (z + depth_scale) / depth_scale * 1.5))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (b, g, r), -1)
            
    return canvas


def make_3d_video_showcase(object_name, total_frames=90, fps=15.0):
    """
    Select the best middle frame, perform detailed segmentation,
    and generate a smooth, slowly rotating 3D volumetric video.
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
        
    # Take the middle frame
    best_frame_file = frame_files[len(frame_files) // 2]
    frame_path = os.path.join(frames_dir, best_frame_file)
    print(f"\nCreating 3D rotating showcase from: {best_frame_file}")
    
    img = cv2.imread(frame_path)
    mask = get_foreground_mask_detailed(img, object_name)
    depth = estimate_depth(img, mask)
    
    video_output_path = os.path.join(obj_dir, f"{object_name}_3d_reconstruction.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (1080, 1080))
    
    # Render rotating frames (360 degrees rotation)
    for idx in range(total_frames):
        angle = (idx / total_frames) * 2 * np.pi
        
        # Render with curved thickness factor = 0.8 for full volume
        rendered_frame = render_3d_frame(
            img, depth, mask, angle=angle, 
            depth_scale=80, thickness_factor=0.8, num_layers=5
        )
        
        out.write(rendered_frame)
        if (idx + 1) % 15 == 0 or idx == 0:
            print(f"  Frame {idx+1}/{total_frames} rendered ({angle/np.pi*180:.1f}°)")
            
    out.release()
    print(f"\n🎉 Detailed 3D Showcase Video saved successfully to:")
    print(f"   👉 {video_output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_3d_video.py <object_name>")
        sys.exit(1)
        
    obj = sys.argv[1]
    make_3d_video_showcase(obj, total_frames=90, fps=15.0)
