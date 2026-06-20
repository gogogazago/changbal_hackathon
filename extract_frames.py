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


def render_single_eye_point_cloud(pixels, depth, mask_resized, dist_map, angle, depth_scale, canvas_w=720, canvas_h=720):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    bg_color = np.array([10, 10, 15], dtype=np.uint8)
    canvas[:] = bg_color
    
    h, w, _ = pixels.shape
    scale_x = 3.2
    scale_y = 2.4
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 50
    
    # Perfect VR stabilization via centroids
    ys, xs = np.where(mask_resized > 0)
    if len(xs) > 0:
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
    else:
        centroid_x = w / 2
        centroid_y = h / 2
        
    points = []
    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask_resized[y, x] > 0:
                d = depth[y, x] * depth_scale
                df = dist_map[y, x]
                
                # Soft blend color near the boundary
                blend = 0.25 + 0.75 * df
                r, g, b = pixels[y, x]
                r_blend = int(r * blend + bg_color[2] * (1.0 - blend))
                g_blend = int(g * blend + bg_color[1] * (1.0 - blend))
                b_blend = int(b * blend + bg_color[0] * (1.0 - blend))
                
                max_t = (0.35 + 0.65 * df) * depth_scale * 1.0
                
                xc = (x - centroid_x) * scale_x
                yc = (y - centroid_y) * scale_y
                
                for layer in range(5):
                    t_offset = (layer / 4.0) * max_t
                    zc = d - t_offset
                    
                    rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                    rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                    rot_y = yc
                    
                    px = int(rot_x + offset_x)
                    py = int(rot_y - rot_z * 0.4 + offset_y)
                    
                    shadow = 1.0 - (layer / 5.0) * 0.45
                    points.append((rot_z, px, py, int(r_blend*shadow), int(g_blend*shadow), int(b_blend*shadow), df))
                    
    points.sort(key=lambda p: p[0])
    
    for z, px, py, r, g, b, df in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size_base = 5.0 + (z + depth_scale) / depth_scale * 2.0
            point_size = max(2, int(size_base * (0.35 + 0.65 * df)))
            cv2.circle(canvas, (px, py), point_size // 2, (b, g, r), -1, lineType=cv2.LINE_AA)
            
    return canvas


def create_3d_point_cloud_image(frame_path, output_path, object_name, depth_scale=85):
    """
    Create a 3D point cloud VR side-by-side stereoscopic image for Google Cardboard SDK.
    """
    from create_3d_video import download_midas_if_needed
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
    pixels = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    if mask_orig is not None:
        mask_resized = cv2.resize(mask_orig, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = np.ones((target_h, target_w), dtype=np.uint8) * 255
        
    h, w, _ = pixels.shape
    
    # Create distance map on mask
    dist_map = cv2.distanceTransform(mask_resized, cv2.DIST_L2, 5)
    dist_map = cv2.normalize(dist_map, None, 0, 1.0, cv2.NORM_MINMAX)
    
    # Estimate depth mapping (use Midas ONNX)
    model_path = download_midas_if_needed()
    net = cv2.dnn.readNet(model_path)
    blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    raw_depth = net.forward()[0, :, :]
    raw_depth_resized = cv2.resize(raw_depth, (target_w, target_h))
    
    # Normalize depth map within object mask
    fg_values = raw_depth_resized[mask_resized > 0]
    depth = np.zeros((target_h, target_w), dtype=np.float32)
    if len(fg_values) > 0:
        min_val = np.min(fg_values)
        max_val = np.max(fg_values)
        depth[mask_resized > 0] = (fg_values - min_val) / (max_val - min_val + 1e-6)
        
    # Render Left eye and Right eye (with stereoscopic separation of 0.03 rad)
    left_eye = render_single_eye_point_cloud(pixels, depth, mask_resized, dist_map, angle=-0.02, depth_scale=depth_scale, canvas_w=720, canvas_h=720)
    right_eye = render_single_eye_point_cloud(pixels, depth, mask_resized, dist_map, angle=0.02, depth_scale=depth_scale, canvas_w=720, canvas_h=720)
    
    # Stack SBS
    sbs = np.hstack([left_eye, right_eye])
    
    # Thin dividing line
    cv2.line(sbs, (720, 0), (720, 720), (50, 50, 60), 2)
    
    cv2.imwrite(output_path, sbs)
    print(f"    SBS VR Point Cloud: {output_path}")



def export_object_mesh_to_vr(object_name, frame_path, output_obj_path, output_png_path):
    """
    Export segmented 3D mesh (OBJ) and transparent texture (PNG) 
    compatible with Unity/Google Cardboard SDK.
    """
    from create_3d_video import download_midas_if_needed
    img = cv2.imread(frame_path)
    h_orig, w_orig, _ = img.shape
    
    # Segment object
    if object_name == "headphone":
        mask = get_headphone_mask(img)
    elif object_name == "banana":
        model_path = download_midas_if_needed()
        net = cv2.dnn.readNet(model_path)
        img_small = cv2.resize(img, (w_orig//2, h_orig//2), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img_small, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
        net.setInput(blob)
        raw_depth = net.forward()[0, :, :]
        raw_depth_orig = cv2.resize(raw_depth, (w_orig, h_orig))
        mask = get_banana_mask(img, raw_depth=raw_depth_orig)
    else:
        mask = None

    if mask is None:
        return

    # Create texture PNG: segmented object on transparent background
    segmented = cv2.bitwise_and(img, img, mask=mask)
    bgra = cv2.cvtColor(segmented, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    cv2.imwrite(output_png_path, bgra)
    print(f"    Saved texture: {output_png_path}")
    
    # Downsample grid for lightweight mobile mesh rendering
    scale = 0.15
    target_w, target_h = int(w_orig * scale), int(h_orig * scale)
    mask_small = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # Estimate depth at target size
    model_path = download_midas_if_needed()
    net = cv2.dnn.readNet(model_path)
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    raw_depth = net.forward()[0, :, :]
    raw_depth_resized = cv2.resize(raw_depth, (target_w, target_h))
    
    # Normalize depth map within mask
    fg_values = raw_depth_resized[mask_small > 0]
    depth = np.zeros((target_h, target_w), dtype=np.float32)
    if len(fg_values) > 0:
        min_v = np.min(fg_values)
        max_v = np.max(fg_values)
        depth[mask_small > 0] = (raw_depth_resized[mask_small > 0] - min_v) / (max_v - min_v + 1e-6)
        
    spacing = 0.02
    depth_scale = 1.2
    
    vertices = []
    uvs = []
    grid_to_index = {}
    vertex_count = 0
    
    for y in range(target_h):
        for x in range(target_w):
            if mask_small[y, x] > 0:
                vertex_count += 1
                grid_to_index[(x, y)] = vertex_count
                
                vx = (x - target_w / 2.0) * spacing
                vy = -(y - target_h / 2.0) * spacing
                vz = depth[y, x] * depth_scale
                
                vertices.append((vx, vy, vz))
                u = x / (target_w - 1.0) if target_w > 1 else 0.0
                v = 1.0 - (y / (target_h - 1.0)) if target_h > 1 else 0.0
                uvs.append((u, v))
                
    faces = []
    for y in range(target_h - 1):
        for x in range(target_w - 1):
            corners = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
            if all(c in grid_to_index for c in corners):
                idx0 = grid_to_index[corners[0]]
                idx1 = grid_to_index[corners[1]]
                idx2 = grid_to_index[corners[2]]
                idx3 = grid_to_index[corners[3]]
                faces.append((idx0, idx1, idx2))
                faces.append((idx1, idx3, idx2))
                
    # Write OBJ
    with open(output_obj_path, "w") as f:
        f.write("# Wavefront OBJ file exported for Google Cardboard VR SDK\n")
        f.write(f"# Object: {object_name}\n")
        f.write(f"mtllib {object_name}.mtl\n\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for tu, tv in uvs:
            f.write(f"vt {tu:.6f} {tv:.6f}\n")
        f.write("\nusemtl Material\n")
        for f0, f1, f2 in faces:
            f.write(f"f {f0}/{f0} {f1}/{f1} {f2}/{f2}\n")
            
    print(f"    Saved mesh OBJ: {output_obj_path} (Vertices: {len(vertices)}, Faces: {len(faces)})")
    
    # Write MTL Material file to auto-link the texture
    mtl_path = output_obj_path.replace(".obj", ".mtl")
    texture_filename = os.path.basename(output_png_path)
    with open(mtl_path, "w") as fm:
        fm.write("# Material File\n")
        fm.write("newmtl Material\n")
        fm.write("Ka 1.0 1.0 1.0\n")
        fm.write("Kd 1.0 1.0 1.0\n")
        fm.write("Ks 0.0 0.0 0.0\n")
        fm.write(f"map_Kd {texture_filename}\n")


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
        
    print(f"\n  📦 Step 5: Exporting OBJ 3D mesh and texture for Google Cardboard VR...")
    first_frame = os.path.join(frames_dir, "frame_0000.png")
    if os.path.exists(first_frame):
        export_object_mesh_to_vr(
            object_name,
            first_frame,
            os.path.join(renders_dir, f"{object_name}.obj"),
            os.path.join(renders_dir, f"{object_name}_texture.png")
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
