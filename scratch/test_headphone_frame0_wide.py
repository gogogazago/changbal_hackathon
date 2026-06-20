import cv2
import numpy as np
import os

def get_headphone_mask_wider(img):
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    # Expanded bounding box to prevent clipping objects touching the sides
    x1, y1 = int(sw * 0.02), int(sh * 0.10)
    x2, y2 = int(sw * 0.98), int(sh * 0.90)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
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

def test_headphone_frame0_wide():
    img_path = "pixel_frames/headphone/frames/frame_0000.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        mask = get_headphone_mask_wider(img)
        segmented = cv2.bitwise_and(img, img, mask=mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_headphone_frame0_wide.png", segmented)
        print("Saved wide segmented headphone frame 0 to scratch/test_headphone_frame0_wide.png")

if __name__ == "__main__":
    test_headphone_frame0_wide()
