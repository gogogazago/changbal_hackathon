import cv2
import numpy as np
import os
import sys

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_3d_video import get_headphone_mask, get_banana_mask

def estimate_ai_depth(img, mask):
    # Load MiDaS ONNX
    net = cv2.dnn.readNet("model-small.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    h, w, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    depth = net.forward()
    depth = depth[0, :, :]
    depth_resized = cv2.resize(depth, (w, h))
    
    # Mask & Normalize to 0-255 within the mask
    fg_values = depth_resized[mask > 0]
    final_depth = np.zeros((h, w), dtype=np.uint8)
    if len(fg_values) > 0:
        min_val = np.min(fg_values)
        max_val = np.max(fg_values)
        normalized_fg = ((fg_values - min_val) / (max_val - min_val + 1e-6) * 255.0).astype(np.uint8)
        final_depth[mask > 0] = normalized_fg
        
    return final_depth

def render_3d_frame_thick(img, depth, mask, angle=0.0, depth_scale=80, thickness_factor=0.6, num_layers=5):
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
    canvas[:] = [10, 10, 15]
    
    scale_x = 4.2
    scale_y = 3.2
    offset_x = canvas_w // 2
    offset_y = canvas_h // 2 - 100
    
    points = []
    
    for y in range(small_h):
        for x in range(small_w):
            if mask_small[y, x] > 0:
                d = (depth_small[y, x] / 255.0) * depth_scale
                
                # Base thickness (0.35) prevents flattening at the boundaries
                max_t = (0.35 + 0.65 * dist_map[y, x]) * depth_scale * thickness_factor
                
                xc = (x - small_w // 2) * scale_x
                yc = (y - small_h // 2) * scale_y
                
                for layer in range(num_layers):
                    t_offset = (layer / max(1, num_layers - 1)) * max_t
                    zc = d - t_offset
                    
                    rot_x = xc * np.cos(angle) + zc * np.sin(angle)
                    rot_z = -xc * np.sin(angle) + zc * np.cos(angle)
                    rot_y = yc
                    
                    px = int(rot_x + offset_x)
                    py = int(rot_y - rot_z * 0.4 + offset_y)
                    
                    shadow = 1.0 - (layer / num_layers) * 0.45
                    b, g, r = img_small[y, x]
                    r_sh = int(r * shadow)
                    g_sh = int(g * shadow)
                    b_sh = int(b * shadow)
                    
                    points.append((rot_z, px, py, r_sh, g_sh, b_sh))
                    
    points.sort(key=lambda p: p[0])
    
    for z, px, py, r, g, b in points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size = max(2, int(3 + (z + depth_scale) / depth_scale * 1.5))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (b, g, r), -1)
            
    return canvas

def test_ai_rendering():
    # Headphone
    hp_img = cv2.imread("pixel_frames/headphone/frames/frame_0140.png")
    hp_mask = get_headphone_mask(hp_img)
    hp_depth = estimate_ai_depth(hp_img, hp_mask)
    
    # Render at 45 degrees to see side profile thickness
    hp_rendered = render_3d_frame_thick(hp_img, hp_depth, hp_mask, angle=0.8, depth_scale=85, thickness_factor=0.8)
    cv2.imwrite("scratch/test_ai_thick_headphone.png", hp_rendered)
    print("Saved thick AI headphone rendering to scratch/test_ai_thick_headphone.png")

    # Banana
    ban_img = cv2.imread("pixel_frames/banana/frames/frame_0000.png")
    ban_mask = get_banana_mask(ban_img)
    ban_depth = estimate_ai_depth(ban_img, ban_mask)
    
    # Render at 45 degrees
    ban_rendered = render_3d_frame_thick(ban_img, ban_depth, ban_mask, angle=0.8, depth_scale=85, thickness_factor=1.0)
    cv2.imwrite("scratch/test_ai_thick_banana.png", ban_rendered)
    print("Saved thick AI banana rendering to scratch/test_ai_thick_banana.png")

if __name__ == "__main__":
    test_ai_rendering()
