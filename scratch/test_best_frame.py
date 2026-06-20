import cv2
import numpy as np
import os

def get_foreground_mask_detailed(img, object_name):
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if object_name == "headphone":
        rect = (int(w * 0.08), int(h * 0.15), int(w * 0.84), int(h * 0.7))
        mask_gc = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT)
            gc_mask = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype('uint8')
        except:
            gc_mask = np.ones((h, w), np.uint8) * 255
            
        _, desk_mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(gc_mask, cv2.bitwise_not(desk_mask))
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        mask_clean = np.zeros_like(mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_clean[labels == largest_label] = 255
            mask = mask_clean
            
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
    elif object_name == "banana":
        # We use the clean mask GrabCut logic here for test
        mask = np.zeros((h, w), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        
        x1, y1 = int(w * 0.12), int(h * 0.22)
        x2, y2 = int(w * 0.88), int(h * 0.68)
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:h, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:w] = cv2.GC_BGD
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 60, 60])
        upper_yellow = np.array([38, 255, 255])
        yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
        
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([135, 255, 255])
        blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
        mask[blue_pixels > 0] = cv2.GC_BGD
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask[(gray > 180) & (mask != cv2.GC_BGD) & (np.arange(w) > w * 0.70)] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img, mask, None, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_MASK)
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
            mask = final_mask
        else:
            mask = bin_mask
    else:
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
    return mask

def test_find_best_frame(object_name):
    frames_dir = f"pixel_frames/{object_name}/frames"
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    
    # Check 5 frames across the sequence
    test_indices = np.linspace(0, len(frame_files) - 1, 5, dtype=int)
    
    best_idx = -1
    best_score = -1
    best_mask = None
    best_img = None
    best_name = ""
    
    for idx in test_indices:
        frame_file = frame_files[idx]
        img = cv2.imread(os.path.join(frames_dir, frame_file))
        h, w, _ = img.shape
        
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = get_foreground_mask_detailed(img_small, object_name)
        
        area = np.sum(mask > 0)
        
        # We penalize masks that touch the bounding box limits (which indicates clipping/occlusion)
        # Check if mask touches left/right bounding box limit
        sh, sw = mask.shape
        touches_left = np.any(mask[:, int(sw * 0.15)] > 0)
        touches_right = np.any(mask[:, int(sw * 0.77)] > 0)
        
        # If banana touches right limit, it means it's overlapping with the bottle/clipped
        score = area
        if object_name == "banana":
            if touches_right:
                score *= 0.1 # Severe penalty for being cut off by bottle
                
        print(f"Frame {frame_file}: Area={area}, TouchesRight={touches_right}, Score={score}")
        
        if score > best_score:
            best_score = score
            best_idx = idx
            best_mask = mask
            best_img = img_small
            best_name = frame_file
            
    print(f"👉 Selected Best Frame: {best_name} (Score: {best_score})")
    segmented = cv2.bitwise_and(best_img, best_img, mask=best_mask)
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite(f"scratch/best_frame_selected_{object_name}.png", segmented)

if __name__ == "__main__":
    print("--- Testing Headphone ---")
    test_find_best_frame("headphone")
    print("\n--- Testing Banana ---")
    test_find_best_frame("banana")
