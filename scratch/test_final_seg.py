import cv2
import numpy as np
import os

def get_headphone_mask(img):
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
    x1, y1 = int(sw * 0.08), int(sh * 0.15)
    x2, y2 = int(sw * 0.92), int(sh * 0.85)
    
    mask[0:y1, :] = cv2.GC_BGD
    mask[y2:sh, :] = cv2.GC_BGD
    mask[:, 0:x1] = cv2.GC_BGD
    mask[:, x2:sw] = cv2.GC_BGD
    
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    mask[(gray < 100) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
    mask[(gray < 65) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
    
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
        
    return cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

def get_banana_mask(img):
    h, w, _ = img.shape
    scale = 0.5
    img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw, _ = img_small.shape
    
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    
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
    
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([135, 255, 255])
    blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
    mask[blue_pixels > 0] = cv2.GC_BGD
    
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    mask[(gray > 180) & (mask != cv2.GC_BGD) & (np.arange(sw) > sw * 0.70)] = cv2.GC_BGD
    
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
        
    return cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)

def test_final_segmentation():
    # Banana - Frame 0 (unoccluded)
    ban_img = cv2.imread("pixel_frames/banana/frames/frame_0000.png")
    ban_mask = get_banana_mask(ban_img)
    ban_seg = cv2.bitwise_and(ban_img, ban_img, mask=ban_mask)
    cv2.imwrite("scratch/final_test_banana.png", ban_seg)
    print("Saved final banana test to scratch/final_test_banana.png")

    # Headphone - Frame 140
    hp_img = cv2.imread("pixel_frames/headphone/frames/frame_0140.png")
    hp_mask = get_headphone_mask(hp_img)
    hp_seg = cv2.bitwise_and(hp_img, hp_img, mask=hp_mask)
    cv2.imwrite("scratch/final_test_headphone.png", hp_seg)
    print("Saved final headphone test to scratch/final_test_headphone.png")

if __name__ == "__main__":
    test_final_segmentation()
