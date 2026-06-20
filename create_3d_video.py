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
    Isolate headphones using mask GrabCut. Ignores wood table orange/brown colors
    using precise Hue-Saturation ranges to avoid clipping neutral headphones.
    """
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD  # Default probably background
    
    # Bounding box
    x1, y1 = int(sw * 0.05), int(sh * 0.12)
    x2, y2 = int(sw * 0.95), int(sh * 0.88)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    # Wood table desk color has Hue in 8-25 and Saturation > 40
    # This prevents neutral grey/black headband reflections from being clipped
    is_wood = (h_channel >= 8) & (h_channel <= 25) & (s_channel > 40)
    mask[is_wood & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
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
    Isolate banana using mask GrabCut. Crops out water bottle, ignores blue colors,
    and filters out desaturated table wood/desk shadows dynamically using distance transform.
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
    
    # Distance transform from yellow pixels
    dist_from_yellow = cv2.distanceTransform(255 - yellow_pixels, cv2.DIST_L2, 5)
    
    # Dynamic hybrid background filter:
    # 1. Any pixel further than 22 pixels from yellow is definite background (GC_BGD)
    # 2. Any non-yellow pixel close to yellow but having low saturation (S < 95) is definite background (GC_BGD)
    s_channel = hsv[:, :, 1]
    mask[(dist_from_yellow > 22) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    mask[(s_channel < 95) & (yellow_pixels == 0) & (dist_from_yellow <= 22) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
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


import urllib.request

def download_midas_if_needed():
    """
    Ensure the MiDaS ONNX model is available locally.
    """
    model_path = "model-small.onnx"
    if not os.path.exists(model_path):
        print("📥 Downloading pre-trained AI depth model (MiDaS v2.1 Small, ~58MB)...")
        model_url = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx"
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Download completed.")
    return model_path


def estimate_depth(img, mask):
    """
    AI Depth Estimation: Runs MiDaS ONNX via OpenCV DNN to estimate continuous 
    monocular relative depth, and normalizes it exclusively within the object boundaries.
    """
    model_path = download_midas_if_needed()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    
    depth = net.forward()
    depth = depth[0, :, :]
    depth_resized = cv2.resize(depth, (w, h))
    
    # Mask & Normalize depth values ONLY within the foreground segment
    fg_values = depth_resized[mask > 0]
    final_depth = np.zeros((h, w), dtype=np.uint8)
    
    if len(fg_values) > 0:
        min_val = np.min(fg_values)
        max_val = np.max(fg_values)
        normalized_fg = ((fg_values - min_val) / (max_val - min_val + 1e-6) * 255.0).astype(np.uint8)
        final_depth[mask > 0] = normalized_fg
        
    return final_depth


def render_3d_frame(img_small, depth_small, mask_small, dist_map, angle=0.0, depth_scale=80, thickness_factor=0.6, num_layers=5, canvas_w=1080, canvas_h=1080):
    """
    Project 3D points with round/cylindrical volume thickness, Y-axis rotation.
    Applies a minimum boundary thickness to prevent flat paper-like edges.
    Operates on pre-resized, pre-smoothed inputs for maximum performance.
    """
    small_h, small_w, _ = img_small.shape
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]  # Background
    
    scale_x = 4.2
    scale_y = 3.2
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 100
    
    points = []
    
    # Optimize by looping only over foreground pixels
    ys, xs = np.where(mask_small > 0)
    
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        d = (depth_small[y, x] / 255.0) * depth_scale
        
        # Base boundary thickness coefficient (0.35) avoids paper-flat shapes
        max_t = (0.35 + 0.65 * dist_map[y, x]) * depth_scale * thickness_factor
        
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
            
            points.append((rot_z, px, py, int(r * shadow), int(g * shadow), int(b * shadow)))
                    
    # Painter's Algorithm
    points.sort(key=lambda p: p[0])
    
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            # Slightly larger splats close empty space gaps between adjacent points
            size = max(4, int(5 + (z + depth_scale) / depth_scale * 2.0))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (b, g, r), -1)
            
    return canvas


def discover_video_path(object_name):
    """
    Find the source video path dynamically under video/<object_name>/.
    Matches the discovery pattern from extract_frames.py.
    """
    video_dir = os.path.join("video", object_name)
    if not os.path.exists(video_dir):
        return None
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    for f in os.listdir(video_dir):
        if f.lower().endswith(video_extensions):
            return os.path.join(video_dir, f)
    return None


def make_3d_video_showcase(object_name, output_fps=20.0):
    """
    Reads frames directly from the source video file on-the-fly,
    processes each with clean segmentations and smoothed depth mapping,
    and writes high-resolution, butter-smooth 3D standard and cardboard VR showcases.
    """
    obj_dir = os.path.join(OUTPUT_BASE, object_name)
    os.makedirs(obj_dir, exist_ok=True)
    
    video_path = discover_video_path(object_name)
    if not video_path:
        print(f"❌ Error: Source video for '{object_name}' not found under video/")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open source video: {video_path}")
        return
        
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configure thickness, sequence range, and step size for each object
    if object_name == "banana":
        thickness_val = 1.0
        step = 4          # Sample every 4th frame -> 600 / 4 = 150 frames
        max_frames = total_source_frames
    elif object_name == "headphone":
        thickness_val = 0.8
        step = 2          # Sample every 2nd frame
        # Capping at frame index 240 to completely avoid the background office chair appearing at the end
        max_frames = min(240, total_source_frames)
    else:
        thickness_val = 0.6
        step = 3
        max_frames = total_source_frames
        
    print(f"\n🎬 Creating Smooth 3D Rotating Videos for '{object_name}'...")
    print(f"   👉 Source Video: {video_path} ({total_source_frames} total frames, processing up to {max_frames})")
    print(f"   👉 Downsampling: every {step}th frame -> smooth {output_fps} fps playback")
    
    # Video Output Paths
    video_std_path = os.path.join(obj_dir, f"{object_name}_3d_reconstruction.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # 1. Standard Writer (1080x1080)
    out_std = cv2.VideoWriter(video_std_path, fourcc, output_fps, (1080, 1080))
    
    frame_count = 0
    saved_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % step == 0:
            # Segment and estimate depth for this specific frame on-the-fly
            if object_name == "headphone":
                mask = get_headphone_mask(frame)
            elif object_name == "banana":
                mask = get_banana_mask(frame)
            else:
                gray_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_tmp, 127, 255, cv2.THRESH_BINARY_INV)
                
            depth = estimate_depth(frame, mask)
            
            # Pre-compute small-resolution buffers ONCE per frame to speed up double/triple rendering
            h, w, _ = frame.shape
            scale = 0.25
            small_w, small_h = int(w * scale), int(h * scale)
            
            img_small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            depth_small = cv2.resize(depth, (small_w, small_h), interpolation=cv2.INTER_AREA)
            depth_small = cv2.GaussianBlur(depth_small, (15, 15), 0)
            mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            
            dist_map = cv2.distanceTransform(mask_small, cv2.DIST_L2, 5)
            dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
            
            # Gentle hover sway animation to make the 3D pop
            sway_angle = 0.08 * np.sin(saved_count * 0.1)
            
            # A. Render Standard Frame
            rendered_std = render_3d_frame(
                img_small, depth_small, mask_small, dist_map, angle=sway_angle, 
                depth_scale=85, thickness_factor=thickness_val, num_layers=5
            )
            out_std.write(rendered_std)
            
            saved_count += 1
            if saved_count % 15 == 0 or saved_count == 1:
                print(f"  Frame {saved_count} written (source index: {frame_count}/{max_frames}, sway={sway_angle/np.pi*180:.1f}°)")
                
        frame_count += 1
        
    cap.release()
    out_std.release()
    
    print(f"\n🎉 Smooth 3D Video saved successfully to:")
    print(f"   👉 Standard:  {video_std_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 create_3d_video.py <object_name>")
        sys.exit(1)
        
    obj = sys.argv[1]
    make_3d_video_showcase(obj, output_fps=20.0)
