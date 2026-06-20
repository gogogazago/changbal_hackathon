import cv2
import numpy as np
import os

def test_masked_midas():
    # Load image and mask
    img_path = "pixel_frames/headphone/frames/frame_0140.png"
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    
    # Generate the headphone mask using our new high-quality function
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from create_3d_video import get_headphone_mask
    mask = get_headphone_mask(img)
    
    # Load MiDaS ONNX
    net = cv2.dnn.readNet("model-small.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Forward pass
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
    net.setInput(blob)
    depth = net.forward()
    depth = depth[0, :, :]
    depth_resized = cv2.resize(depth, (w, h))
    
    # Mask it first
    masked_depth = np.zeros_like(depth_resized)
    masked_depth[mask > 0] = depth_resized[mask > 0]
    
    # Normalize ONLY the foreground region to 0-255
    fg_values = depth_resized[mask > 0]
    if len(fg_values) > 0:
        min_val = np.min(fg_values)
        max_val = np.max(fg_values)
        print(f"MiDaS FG range: min={min_val}, max={max_val}")
        
        # Linear normalization to 0-255
        normalized_fg = ((fg_values - min_val) / (max_val - min_val + 1e-6) * 255.0).astype(np.uint8)
        
        final_depth = np.zeros((h, w), dtype=np.uint8)
        final_depth[mask > 0] = normalized_fg
    else:
        final_depth = np.zeros((h, w), dtype=np.uint8)
        
    os.makedirs("scratch", exist_ok=True)
    cv2.imwrite("scratch/test_midas_masked_depth.png", final_depth)
    print("Saved masked MiDaS depth to scratch/test_midas_masked_depth.png")

if __name__ == "__main__":
    test_masked_midas()
