import cv2
import numpy as np
import os

def test_banana_hybrid():
    frames = ["frame_0000.png", "frame_0200.png", "frame_0400.png"]
    os.makedirs("scratch", exist_ok=True)
    
    for f in frames:
        img_path = f"pixel_frames/banana/frames/{f}"
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        mask = np.zeros((sh, sw), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 60, 60])
        upper_yellow = np.array([38, 255, 255])
        yellow_pixels = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Distance transform from yellow pixels
        # 255 - yellow_pixels makes yellow pixels 0, background 255
        dist_from_yellow = cv2.distanceTransform(255 - yellow_pixels, cv2.DIST_L2, 5)
        
        # Bounding box
        x1, y1 = int(sw * 0.12), int(sh * 0.22)
        x2, y2 = int(sw * 0.88), int(sh * 0.68)
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:sh, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:sw] = cv2.GC_BGD
        
        # Mark yellow pixels as foreground
        mask[(yellow_pixels > 0) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
        
        # Dynamic hybrid background filter:
        # 1. Any pixel further than 22 pixels from yellow is definite background (GC_BGD)
        # 2. Any pixel close to yellow but having low saturation (S < 95) is definite background (GC_BGD)
        s_channel = hsv[:, :, 1]
        mask[(dist_from_yellow > 22) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
        mask[(s_channel < 95) & (dist_from_yellow <= 22) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
        
        # Blue bottle filter
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([135, 255, 255])
        blue_pixels = cv2.inRange(hsv, lower_blue, upper_blue)
        mask[blue_pixels > 0] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
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
        else:
            final_mask = bin_mask
            
        final_mask_orig = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        segmented = cv2.bitwise_and(img, img, mask=final_mask_orig)
        
        out_path = f"scratch/test_banana_hybrid_{f}"
        cv2.imwrite(out_path, segmented)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    test_banana_hybrid()
