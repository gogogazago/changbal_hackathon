import cv2
import numpy as np
import os

def get_foreground_mask(img, object_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if object_name == "headphone":
        _, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            outer_mask = np.zeros_like(thresh)
            cv2.drawContours(outer_mask, [contours[0]], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, outer_mask)
    elif object_name == "banana":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 70, 70])
        upper_yellow = np.array([38, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    else:
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return mask

def estimate_depth(img, mask):
    # Standard blended depth map
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
    
    blended = cv2.addWeighted(dist_norm, 0.7, grad_norm, 0.3, 0)
    return cv2.bitwise_and(blended, blended, mask=mask)

def test_multi_frame_fusion(object_name, sweep_angle_deg=180):
    frames_dir = f"pixel_frames/{object_name}/frames"
    if not os.path.exists(frames_dir):
        print("Frames dir not found")
        return
        
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    if not frame_files:
        print("No frames found")
        return
        
    print(f"Fusing {len(frame_files)} frames for '{object_name}'...")
    
    # We will sample 10 frames evenly across the sequence
    sample_indices = np.linspace(0, len(frame_files) - 1, 10, dtype=int)
    
    world_points = []
    
    # Scale parameters
    scale_x = 4.2
    scale_y = 3.2
    depth_scale = 80
    
    sweep_angle = np.radians(sweep_angle_deg)
    
    for i, idx in enumerate(sample_indices):
        frame_file = frame_files[idx]
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        h, w, _ = img.shape
        
        # Downsample for processing
        scale = 0.25
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = get_foreground_mask(img_small, object_name)
        depth = estimate_depth(img_small, mask)
        
        sh, sw = mask.shape
        
        # Calculate orbit angle for this frame
        # Camera is rotating around the object
        angle = (i / (len(sample_indices) - 1)) * sweep_angle - (sweep_angle / 2)
        
        for y in range(sh):
            for x in range(sw):
                if mask[y, x] > 0:
                    d = (depth[y, x] / 255.0) * depth_scale
                    
                    # Centered camera coordinates
                    xc = (x - sw // 2) * scale_x
                    yc = (y - sh // 2) * scale_y
                    zc = d - (depth_scale / 2) # Center depth around origin
                    
                    # Back-project: Rotate camera coordinate to world space by -angle
                    # (since camera moved by +angle, the object points in camera space must rotate by -angle to align in world space)
                    rot_angle = -angle
                    xw = xc * np.cos(rot_angle) + zc * np.sin(rot_angle)
                    zw = -xc * np.sin(rot_angle) + zc * np.cos(rot_angle)
                    yw = yc
                    
                    b, g, r = img_small[y, x]
                    world_points.append((xw, yw, zw, r, g, b))
                    
    print(f"Total accumulated points: {len(world_points)}")
    
    # Render the accumulated cloud from a fixed angle to see how it looks
    canvas_w, canvas_h = 1080, 1080
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]
    
    # Sort points by Z in world coordinates (painters algorithm for rendering)
    # Let's render from a 45 degree angle
    render_angle = np.radians(45)
    render_points = []
    
    for xw, yw, zw, r, g, b in world_points:
        # Rotate point by render_angle to simulate virtual camera looking at it
        rx = xw * np.cos(render_angle) + zw * np.sin(render_angle)
        rz = -xw * np.sin(render_angle) + zw * np.cos(render_angle)
        ry = yw
        
        px = int(rx + canvas_w // 2)
        py = int(ry - rz * 0.4 + canvas_h // 2 - 100)
        
        render_points.append((rz, px, py, r, g, b))
        
    render_points.sort(key=lambda p: p[0])
    
    for rz, px, py, r, g, b in render_points:
        if 0 <= px < canvas_w and 0 <= py < canvas_h:
            size = max(2, int(3 + (rz + depth_scale) / depth_scale * 1.5))
            cv2.rectangle(canvas, (px - size//2, py - size//2), (px + size//2, py + size//2), (int(b), int(g), int(r)), -1)
            
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite(f"scratch/test_fusion_{object_name}.png", canvas)
    print(f"Saved fused render to scratch/test_fusion_{object_name}.png")

if __name__ == "__main__":
    test_multi_frame_fusion("headphone", 120)
    test_multi_frame_fusion("banana", 180)
