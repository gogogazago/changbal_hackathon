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
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] > 350:
            final_mask[labels == idx] = 255
            
    return cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def get_banana_mask(img, raw_depth=None):
    """
    Isolate banana and water bottle using mask GrabCut guided by color range
    and 3D plane-fitting depth differences to remove wood table backgrounds cleanly.
    """
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    # If raw_depth is not provided, estimate it on-the-fly (backward compatibility)
    if raw_depth is None:
        model_path = download_midas_if_needed()
        net = cv2.dnn.readNet(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        blob = cv2.dnn.blobFromImage(img_small, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
        net.setInput(blob)
        raw_depth_small = net.forward()[0, :, :]
        raw_depth_small = cv2.resize(raw_depth_small, (sw, sh))
    else:
        raw_depth_small = cv2.resize(raw_depth, (sw, sh))
        
    # Fit plane to table background
    y_indices, x_indices = np.mgrid[0:sh, 0:sw]
    is_margin = (x_indices < int(sw * 0.15)) | (x_indices > int(sw * 0.85))
    is_row_range = (y_indices > int(sh * 0.18)) & (y_indices < int(sh * 0.80))
    table_fit_mask = is_margin & is_row_range
    
    X_fit = x_indices[table_fit_mask]
    Y_fit = y_indices[table_fit_mask]
    Z_fit = raw_depth_small[table_fit_mask]
    
    A = np.column_stack((X_fit, Y_fit, np.ones_like(X_fit)))
    plane_params, _, _, _ = np.linalg.lstsq(A, Z_fit, rcond=None)
    a, b, c = plane_params
    
    predicted_table_depth = a * x_indices + b * y_indices + c
    depth_diff = raw_depth_small - predicted_table_depth
    
    bbox = (y_indices > int(sh * 0.10)) & (y_indices < int(sh * 0.85)) & (x_indices > int(sw * 0.05)) & (x_indices < int(sw * 0.95))
    max_diff = np.max(depth_diff[bbox]) if np.any(bbox) else 1.0
    normalized_diff = np.zeros_like(depth_diff)
    normalized_diff[depth_diff > 0] = depth_diff[depth_diff > 0] / (max_diff + 1e-6)
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    # Bounding box: crop left and top margins slightly, open on the right
    x1, y1 = int(sw * 0.08), int(sh * 0.12)
    x2, y2 = sw, int(sh * 0.88)
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    # Yellow banana (GC_FGD)
    lower_yellow = np.array([12, 90, 50])
    upper_yellow = np.array([38, 255, 255])
    yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    # Blue water bottle label/cap (GC_FGD) - highly selective
    lower_blue = np.array([90, 110, 50])
    upper_blue = np.array([135, 255, 255])
    blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
    mask[(blue_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    # High depth difference pixels (definite foreground seeds, e.g. center of the bottle)
    mask[(normalized_diff > 0.45) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    # Probable Foreground
    mask[(normalized_diff > 0.15) & (mask != cv2.GC_FGD) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
    
    # Wood table color
    is_wood = (h_channel >= 8) & (h_channel <= 25) & (s_channel > 30) & (s_channel < 85)
    
    # Definite Background: flat wood
    is_table_wood = is_wood & (normalized_diff < 0.20) & (yellow_pixels == 0) & (blue_pixels == 0)
    mask[is_table_wood & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
    # Flat paper card/white desk
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    is_flat_white = (gray > 165) & (s_channel < 35) & (normalized_diff < 0.20) & (yellow_pixels == 0) & (blue_pixels == 0)
    mask[is_flat_white & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
    # Low depth diff overall
    is_at_or_below_table = (normalized_diff < 0.08)
    mask[is_at_or_below_table & (mask != cv2.GC_FGD) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    except:
        bin_mask = yellow_pixels | blue_pixels
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    
    # Connected components filter: keep components that contain yellow (banana) or blue (bottle)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
    final_mask = np.zeros_like(bin_mask)
    for idx in range(1, num_labels):
        comp_mask = (labels == idx)
        has_yellow = np.sum(comp_mask & (yellow_pixels > 0)) > 8
        has_blue = np.sum(comp_mask & (blue_pixels > 0)) > 8
        
        if (has_yellow or has_blue) and stats[idx, cv2.CC_STAT_AREA] > 300:
            final_mask[comp_mask] = 255
            
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



def render_3d_frame(img_small, depth_small, mask_small, dist_map, angle=0.0, depth_scale=80, thickness_factor=0.6, num_layers=5, canvas_w=1080, canvas_h=1080):
    """
    Project 3D points with round/cylindrical volume thickness, Y-axis rotation.
    Applies a feathered circular splat scaling and color blending to make boundaries soft.
    Operates on pre-resized, pre-smoothed inputs for maximum performance.
    """
    small_h, small_w, _ = img_small.shape
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    bg_color = np.array([10, 10, 15], dtype=np.uint8)
    canvas[:] = bg_color
    
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
        
        # Distance to boundary (0.0 at edge, 1.0 at center)
        dist_factor = dist_map[y, x]
        
        # Soft blend color near the boundary to prevent hard edges
        blend = 0.25 + 0.75 * dist_factor
        
        b, g, r = img_small[y, x]
        r_blend = int(r * blend + bg_color[2] * (1.0 - blend))
        g_blend = int(g * blend + bg_color[1] * (1.0 - blend))
        b_blend = int(b * blend + bg_color[0] * (1.0 - blend))
        
        # Base boundary thickness coefficient (0.35) avoids paper-flat shapes
        max_t = (0.35 + 0.65 * dist_factor) * depth_scale * thickness_factor
        
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
            
            rc = int(r_blend * shadow)
            gc = int(g_blend * shadow)
            bc = int(b_blend * shadow)
            
            points.append((rot_z, px, py, rc, gc, bc, dist_factor))
                    
    # Painter's Algorithm
    points.sort(key=lambda p: p[0])
    
    for z, px, py, r, g, b, df in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            # Scale point size based on distance map to make boundary points smaller
            size_base = 5.0 + (z + depth_scale) / depth_scale * 2.0
            point_size = max(2, int(size_base * (0.35 + 0.65 * df)))
            
            # Use anti-aliased circle instead of blocky rectangle for smooth edges
            cv2.circle(canvas, (px, py), point_size // 2, (b, g, r), -1, lineType=cv2.LINE_AA)
            
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
    and writes high-resolution, butter-smooth 3D standard rotating showcases.
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
    
    # Load MiDaS network ONCE outside the loop to optimize performance
    model_path = download_midas_if_needed()
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
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
            h, w, _ = frame.shape
            
            # 1. Run raw depth estimation once
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
            net.setInput(blob)
            raw_depth = net.forward()[0, :, :]
            raw_depth_orig = cv2.resize(raw_depth, (w, h))
            
            # 2. Get segment mask
            if object_name == "headphone":
                mask = get_headphone_mask(frame)
            elif object_name == "banana":
                mask = get_banana_mask(frame, raw_depth=raw_depth_orig)
            else:
                gray_tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_tmp, 127, 255, cv2.THRESH_BINARY_INV)
                
            # 3. Mask & Normalize depth values ONLY within the foreground segment
            fg_values = raw_depth_orig[mask > 0]
            depth = np.zeros((h, w), dtype=np.uint8)
            if len(fg_values) > 0:
                min_val = np.min(fg_values)
                max_val = np.max(fg_values)
                normalized_fg = ((fg_values - min_val) / (max_val - min_val + 1e-6) * 255.0).astype(np.uint8)
                depth[mask > 0] = normalized_fg
            
            # Pre-compute small-resolution buffers ONCE per frame
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
            
            # A. Render Standard Frame (feathered circular splats)
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
