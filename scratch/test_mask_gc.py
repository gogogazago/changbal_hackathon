import cv2
import numpy as np
import os

def test_mask_grabcut():
    hp_frame = "pixel_frames/headphone/frames/frame_0140.png"
    if os.path.exists(hp_frame):
        img = cv2.imread(hp_frame)
        h, w, _ = img.shape
        
        scale = 0.5
        img_small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh, sw, _ = img_small.shape
        
        # Initialize GrabCut mask
        mask = np.zeros((sh, sw), np.uint8)
        mask[:] = cv2.GC_PR_BGD  # Default to probably background
        
        # Define bounding box where headphones are
        x1, y1 = int(sw * 0.08), int(sh * 0.15)
        x2, y2 = int(sw * 0.92), int(sh * 0.85)
        
        # Outside bounding box is definitely background
        mask[0:y1, :] = cv2.GC_BGD
        mask[y2:sh, :] = cv2.GC_BGD
        mask[:, 0:x1] = cv2.GC_BGD
        mask[:, x2:sw] = cv2.GC_BGD
        
        # Inside bounding box, classify by color/brightness
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Dark pixels are probably foreground
        mask[(gray < 100) & (mask != cv2.GC_BGD)] = cv2.GC_PR_FGD
        # Very dark pixels are definitely foreground
        mask[(gray < 65) & (mask != cv2.GC_BGD)] = cv2.GC_FGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Run GrabCut
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # Extract foreground mask
        bin_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        
        # Keep only largest component
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask)
        final_mask = np.zeros_like(bin_mask)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask[labels == largest_label] = 255
        else:
            final_mask = bin_mask
            
        segmented = cv2.bitwise_and(img_small, img_small, mask=final_mask)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_mask_grabcut.png", segmented)
        cv2.imwrite("scratch/test_mask_grabcut_mask.png", final_mask)
        print("Saved mask GrabCut test to scratch/test_mask_grabcut.png")

if __name__ == "__main__":
    test_mask_grabcut()
