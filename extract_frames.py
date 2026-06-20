#!/usr/bin/env python3
"""
Frame Extraction Script for 3D Reconstruction Pipeline
Automatically discovers object folders under video/ and creates
matching output structure under pixel_frames/<object_name>/.
"""

import cv2
import os
import sys
import numpy as np
from PIL import Image

VIDEO_DIR = "video"
OUTPUT_BASE = "pixel_frames"
# Extract every Nth frame (to get ~20-30 key frames from 274 total)
FRAME_INTERVAL = 10  


def find_video_file(directory):
    """Find the first video file in a directory."""
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    for f in os.listdir(directory):
        if f.lower().endswith(video_extensions):
            return os.path.join(directory, f)
    return None


def discover_objects():
    """
    Discover object folders under video/.
    Each subfolder (e.g. video/headphone/) is treated as one object.
    Returns a list of (object_name, video_path) tuples.
    """
    objects = []
    for entry in sorted(os.listdir(VIDEO_DIR)):
        obj_dir = os.path.join(VIDEO_DIR, entry)
        if os.path.isdir(obj_dir) and not entry.startswith('.'):
            video_path = find_video_file(obj_dir)
            if video_path:
                objects.append((entry, video_path))
            else:
                print(f"  ⚠  Skipping '{entry}/' — no video file found")
    return objects


def extract_frames(video_path, output_dir):
    """Extract frames from video at regular intervals."""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video Info: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    print(f"  Extracting every {FRAME_INTERVAL}th frame...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % FRAME_INTERVAL == 0:
            filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"    Saved: {filename} ({frame.shape[1]}x{frame.shape[0]})")
        
        frame_count += 1
    
    cap.release()
    print(f"  Extracted {saved_count} frames to '{output_dir}/'")
    return saved_count


def create_pixel_grid(frame_path, output_path, grid_size=64):
    """
    Create a pixel-by-pixel visualization of a frame,
    showing individual pixel blocks at an enlarged scale.
    """
    img = Image.open(frame_path)
    # Downscale to grid_size x grid_size to get pixel blocks
    small = img.resize((grid_size, grid_size), Image.Resampling.NEAREST)
    # Scale back up with nearest-neighbor to show pixel grid
    pixel_art = small.resize((grid_size * 8, grid_size * 8), Image.Resampling.NEAREST)
    pixel_art.save(output_path)
    print(f"    Pixel grid: {output_path}")

from create_3d_video import get_headphone_mask, get_banana_mask

def create_depth_map_visualization(frame_path, output_path, object_name):
    """
    Create a simulated depth map from a single frame using gradient-based estimation,
    masked to isolate only the target object.
    """
    img = cv2.imread(frame_path)
    h, w, _ = img.shape
    
    # Segment foreground
    if object_name == "headphone":
        mask = get_headphone_mask(img)
    elif object_name == "banana":
        mask = get_banana_mask(img)
    else:
        mask = None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    depth = cv2.GaussianBlur(magnitude.astype(np.uint8), (21, 21), 0)
    
    # Mask background to black
    if mask is not None:
        depth = cv2.bitwise_and(depth, depth, mask=mask)
        
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    if mask is not None:
        depth_colored[mask == 0] = [10, 10, 15] # Dark background instead of inferno black
        
    cv2.imwrite(output_path, depth_colored)
    print(f"    Depth map: {output_path}")


def create_3d_point_cloud_image(frame_path, output_path, object_name, depth_scale=50):
    """
    Create a 3D point cloud visualization from a frame, isolating the target object.
    Applies feathered circular splat scaling and color blending to make boundaries soft.
    """
    img_bgr = cv2.imread(frame_path)
    h_orig, w_orig, _ = img_bgr.shape
    
    # Get mask on original resolution
    if object_name == "headphone":
        mask_orig = get_headphone_mask(img_bgr)
    elif object_name == "banana":
        mask_orig = get_banana_mask(img_bgr)
    else:
        mask_orig = None
        
    # Downscale for performance
    target_w, target_h = 160, 284
    img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pixels = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # PIL compatibility
    
    if mask_orig is not None:
        mask_resized = cv2.resize(mask_orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
    h, w, _ = pixels.shape
    
    # Create distance map on mask
    dist_map = cv2.distanceTransform(mask_resized, cv2.DIST_L2, 5)
    dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Create simulated depth using image luminance (brighter = closer)
    gray = np.mean(pixels, axis=2)
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    center_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    center_factor = 1.0 - (center_dist / center_dist.max())
    
    depth = (gray / 255.0 * 0.6 + center_factor * 0.4)
    depth = cv2.GaussianBlur(depth.astype(np.float32), (11, 11), 0)
    
    # Create isometric 3D projection
    canvas_w, canvas_h = 800, 900
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    bg_color = np.array([15, 15, 25], dtype=np.uint8)  # Dark BGR background
    canvas[:] = bg_color
    
    angle = 0.5  # Rotation angle
    scale_x = 3.0
    scale_y = 2.0
    offset_x = canvas_w // 2
    offset_y = 100
    
    points = []
    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask_resized[y, x] > 0:
                d = depth[y, x] * depth_scale
                
                # Distance map factor
                df = dist_map[y, x]
                
                # Soft blend color near the boundary
                blend = 0.25 + 0.75 * df
                r, g, b = pixels[y, x]
                r_blend = int(r * blend + bg_color[2] * (1.0 - blend))
                g_blend = int(g * blend + bg_color[1] * (1.0 - blend))
                b_blend = int(b * blend + bg_color[0] * (1.0 - blend))
                
                iso_x = (x - w//2) * scale_x * np.cos(angle) - d * np.sin(angle) + offset_x
                iso_y = (y) * scale_y + (x - w//2) * scale_x * np.sin(angle) * 0.3 - d * np.cos(angle) * 0.5 + offset_y
                points.append((d, int(iso_x), int(iso_y), r_blend, g_blend, b_blend, df))
                
    points.sort(key=lambda p: p[0])
    
    for d, px, py, r, g, b, df in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size_base = 5.0 + d / depth_scale * 2.0
            point_size = max(2, int(size_base * (0.35 + 0.65 * df)))
            cv2.circle(canvas, (px, py), point_size // 2, (b, g, r), -1, lineType=cv2.LINE_AA)
            
    cv2.putText(canvas, f"3D Point Cloud: {object_name.capitalize()}", (50, canvas_h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
                
    cv2.imwrite(output_path, canvas)
    print(f"    3D point cloud: {output_path}")


def create_stereo_pair(frame_path, output_path, object_name, disparity=20):
    """
    Create a stereoscopic side-by-side view for Google Cardboard compatibility,
    isolating only the target object.
    """
    img = cv2.imread(frame_path)
    h, w, _ = img.shape
    
    if object_name == "headphone":
        mask = get_headphone_mask(img)
    elif object_name == "banana":
        mask = get_banana_mask(img)
    else:
        mask = None
        
    if mask is not None:
        img_segmented = cv2.bitwise_and(img, img, mask=mask)
        # Set background to dark charcoal instead of pure black
        img_segmented[mask == 0] = [10, 10, 15]
    else:
        img_segmented = img.copy()
        
    eye_w = w // 2
    img_resized = cv2.resize(img_segmented, (eye_w, h))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth = cv2.GaussianBlur(gray, (31, 31), 0)
    
    left_eye = img_resized.copy()
    right_eye = img_resized.copy()
    
    for y in range(h):
        for x in range(eye_w):
            shift = int(depth[y, x] * disparity)
            src_x = min(eye_w - 1, max(0, x + shift))
            right_eye[y, x] = img_resized[y, src_x]
            src_x = min(eye_w - 1, max(0, x - shift))
            left_eye[y, x] = img_resized[y, src_x]
            
    stereo = np.hstack([left_eye, right_eye])
    cv2.imwrite(output_path, stereo)
    print(f"    Stereo pair: {output_path}")


def process_object(object_name, video_path):
    """
    Process a single object: extract frames and generate all visualizations.
    Output goes to pixel_frames/<object_name>/ with subfolders.
    """
    obj_output_dir = os.path.join(OUTPUT_BASE, object_name)
    frames_dir = os.path.join(obj_output_dir, "frames")
    pixel_grid_dir = os.path.join(obj_output_dir, "pixel_grid")
    depth_maps_dir = os.path.join(obj_output_dir, "depth_maps")
    renders_dir = os.path.join(obj_output_dir, "3d_renders")

    # Clear and recreate output subdirectories to prevent stale files from lingering
    import shutil
    for d in [frames_dir, pixel_grid_dir, depth_maps_dir, renders_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # Step 1: Extract frames
    print(f"\n  📹 Step 1: Extracting frames from '{video_path}'...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video Info: {width}x{height}, {fps:.1f}fps, {total_frames} frames")
    max_frames = 240 if object_name == "headphone" else total_frames
    print(f"  Extracting every {FRAME_INTERVAL}th frame up to frame {max_frames}...")

    frame_count = 0
    saved_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            filename = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"    Saved: {filename} ({frame.shape[1]}x{frame.shape[0]})")

        frame_count += 1

    cap.release()
    print(f"  ✅ Extracted {saved_count} frames to '{frames_dir}/'")

    # Step 2: Generate visualizations for all key frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])

    print(f"\n  🎨 Step 2: Creating pixel-by-pixel visualizations for {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        base_name = frame_file.replace(".png", "")

        create_pixel_grid(
            frame_path,
            os.path.join(pixel_grid_dir, f"{base_name}_pixels.png")
        )

    print(f"\n  🗺️  Step 3: Creating depth maps for {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        base_name = frame_file.replace(".png", "")

        create_depth_map_visualization(
            frame_path,
            os.path.join(depth_maps_dir, f"{base_name}_depth.png"),
            object_name
        )

    print(f"\n  🧊 Step 4: Creating 3D point cloud renders for {len(frame_files)} frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        base_name = frame_file.replace(".png", "")

        create_3d_point_cloud_image(
            frame_path,
            os.path.join(renders_dir, f"{base_name}_3d.png"),
            object_name
        )

    print(f"\n  ✅ All outputs saved to '{obj_output_dir}/'")


if __name__ == "__main__":
    print("=" * 60)
    print("  3D Reconstruction Pipeline - Frame Extraction")
    print("=" * 60)
    
    # Discover all object folders under video/
    print(f"\n🔍 Scanning '{VIDEO_DIR}/' for object folders...")
    objects = discover_objects()
    
    if not objects:
        print(f"  ❌ No object folders found under '{VIDEO_DIR}/'!")
        print(f"  Expected structure: {VIDEO_DIR}/<object_name>/<video_file>.mp4")
        sys.exit(1)
    
    print(f"  Found {len(objects)} object(s): {', '.join(name for name, _ in objects)}")
    
    # Process each object
    for object_name, video_path in objects:
        print("\n" + "─" * 60)
        print(f"  📦 Processing object: {object_name}")
        print("─" * 60)
        process_object(object_name, video_path)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("  🎉 Pipeline Complete!")
    print("=" * 60)
    print(f"\nOutput Structure:")
    print(f"  {OUTPUT_BASE}/")
    for object_name, _ in objects:
        print(f"    └── {object_name}/")
        print(f"        ├── frames/            (extracted video frames)")
        print(f"        ├── pixel_grid/         (pixel-by-pixel visualizations)")
        print(f"        ├── depth_maps/         (depth map estimations)")
        print(f"        └── 3d_renders/         (3D point cloud renders + stereo)")
