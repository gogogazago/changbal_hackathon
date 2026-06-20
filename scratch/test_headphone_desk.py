import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_headphone_desk_mask():
    frames = ["frame_0100.png", "frame_0220.png"]
    os.makedirs("scratch", exist_ok=True)
    
    for f in frames:
        img_path = f"pixel_frames/headphone/frames/{f}"
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        mask = np.zeros((sh, sw), np.uint8)
        mask[:] = cv2.GC_PR_BGD
        
        # Bounding box
        x1, y1 = int(sw * 0.05), int(sh * 0.12)
        x2, y2 = int(sw * 0.95), int(sh * 0.88)
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:sh, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:sw] = cv2.GC_BGD
        
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]
        
        # Wood table desk color has higher saturation (S > 45)
        # Headphones have very low saturation (neutral grey/black, S < 30)
        # So we mark S > 45 as definite background (GC_BGD)
        mask[(s_channel > 45) & (mask != cv2.GC_BGD)] = cv2.GC_BGD
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        # Mark dark pixels (headphones) as probable foreground (GC_PR_FGD) instead of definite foreground (GC_FGD)
        # to let GrabCut decide based on context and texture
        mask[(gray < 90) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
        mask[(gray < 55) & (mask != cv2.GC_BGD)] = cv2.GC_FGD  # Only extremely dark pixels are definite foreground
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        except:
            bin_mask = np.where(gray < 90, 255, 0).astype('uint8')
            
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
        
        out_path = f"scratch/test_headphone_clean_{f}"
        cv2.imwrite(out_path, segmented)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    test_headphone_desk_mask()
