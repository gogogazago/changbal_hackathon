import cv2
import numpy as np
import os

def get_foreground_mask_cleaned(img, object_name):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if object_name == "headphone":
        # Segment dark headphone
        _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Keep ONLY the single largest connected component (removes desk spots)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        mask = np.zeros_like(thresh)
        if num_labels > 1:
            # Stats format: [x, y, w, h, area]
            # Label 0 is background, so find largest foreground area
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[labels == largest_label] = 255
            
    elif object_name == "banana":
        # Segment yellow
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 70, 70])
        upper_yellow = np.array([38, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep ONLY the single largest connected component (removes water bottle labels, etc)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        mask = np.zeros_like(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask[labels == largest_label] = 255
    else:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        mask = thresh
        
    return mask

def estimate_depth(img, mask):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_norm = cv2.GaussianBlur(grad_norm, (15, 15), 0)
    
    blended = cv2.addWeighted(dist_norm, 0.8, grad_norm, 0.2, 0)
    return cv2.bitwise_and(blended, blended, mask=mask)

def test_multi_frame_fusion_centered(object_name, sweep_angle_deg=120):
    frames_dir = f"pixel_frames/{object_name}/frames"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    
    # We will sample 8 key frames across the sequence
    sample_indices = np.linspace(0, len(frame_files) - 1, 8, dtype=int)
    
    world_points = []
    scale_x = 4.2
    scale_y = 3.2
    depth_scale = 80
    
    sweep_angle = np.radians(sweep_angle_deg)
    
    for i, idx in enumerate(sample_indices):
        frame_file = frame_files[idx]
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        
        scale = 0.25
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = get_foreground_mask_cleaned(img_small, object_name)
        depth = estimate_depth(img_small, mask)
        
        sh, sw = mask.shape
        
        # Calculate Centroid of the mask in 2D image coordinates
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = sw / 2, sh / 2
            
        # Calculate mean depth of the object to center Z-axis
        masked_depths = depth[mask > 0]
        mean_d = np.mean(masked_depths) if len(masked_depths) > 0 else 127
        
        # Calculate orbit angle
        angle = (i / (len(sample_indices) - 1)) * sweep_angle - (sweep_angle / 2)
        
        for y in range(sh):
            for x in range(sw):
                if mask[y, x] > 0:
                    d = depth[y, x]
                    
                    # Project points RELATIVE to the object's 3D centroid
                    # This centers the object at (0, 0, 0) in world space
                    xc = (x - cx) * scale_x
                    yc = (y - cy) * scale_y
                    
                    # Bulge depth goes forwards/backwards from center of volume
                    zc = (d - mean_d) / 255.0 * depth_scale
                    
                    # Back-project: Rotate around Y axis to transform camera space to world space
                    rot_angle = -angle
                    xw = xc * np.cos(rot_angle) + zc * np.sin(rot_angle)
                    zw = -xc * np.sin(rot_angle) + zc * np.cos(rot_angle)
                    yw = yc
                    
                    b, g, r = img_small[y, x]
                    world_points.append((xw, yw, zw, r, g, b))
                    
    # Render accumulated point cloud
    canvas_w, canvas_h = 1080, 1080
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = [10, 10, 15]
    
    render_angle = np.radians(45)
    render_points = []
    
    for xw, yw, zw, r, g, b in world_points:
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
    cv2.imwrite(f"scratch/test_fusion_centered_{object_name}.png", canvas)
    print(f"Saved centered fused render to scratch/test_fusion_centered_{object_name}.png")

if __name__ == "__main__":
    test_multi_frame_fusion_centered("headphone", 120)
    test_multi_frame_fusion_centered("banana", 120)
