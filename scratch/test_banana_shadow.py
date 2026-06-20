import cv2
import numpy as np
import os

def test_banana_shadow():
    img_path = "pixel_frames/banana/frames/frame_0000.png"
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    # We want to refine the mask GrabCut initialization to exclude the table shadow
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    # Crop bounding box tight around the banana
    x1, y1 = int(sw * 0.15), int(sh * 0.25)
    x2, y2 = int(sw * 0.85), int(sh * 0.70)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    
    # Yellow range (narrowed down to exclude grey/brown shadow)
    # Hue: 15 to 35 (yellow)
    # Saturation: 80 to 255 (excludes low-saturation shadows/desk)
    # Value: 80 to 255 (excludes dark shadows)
    lower_yellow = np.array([14, 80, 80])
    upper_yellow = np.array([36, 255, 255])
    yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
    # Explicitly mark low saturation colors (desk shadow/table) as definitely background
    # Restrict to the lower portion (y > sh * 0.52) to preserve the stem and upper banana neck.
    s_channel = hsv[:, :, 1]
    y_coords, x_coords = np.indices((sh, sw))
    mask[(s_channel < 95) & (y_coords > sh * 0.52) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
    
    # Exclude blue pixels (water bottle label)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
    mask[blue_pixels > 0] = cv2.GC_BGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    
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
        
    segmented = cv2.bitwise_and(img_small, img_small, mask=final_mask)
    
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite("scratch/test_banana_shadow_removed.png", segmented)
    print("Saved shadow-removed banana to scratch/test_banana_shadow_removed.png")

if __name__ == "__main__":
    test_banana_shadow()
