import cv2
import urllib.request
import os
import numpy as np

def download_midas():
    model_url = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx"
    model_path = "model-small.onnx"
    if not os.path.exists(model_path):
        print("Downloading MiDaS ONNX model (~58MB)...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    else:
        print("MiDaS ONNX model already exists.")
    return model_path

def test_midas_depth():
    model_path = download_midas()
    
    # Load network
    net = cv2.dnn.readNet(model_path)
    if net.empty():
        print("Error: Could not load DNN network.")
        return
        
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("Successfully loaded MiDaS ONNX model into OpenCV DNN!")
    
    # Let's test on a headphone frame
    img_path = "pixel_frames/headphone/frames/frame_0140.png"
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        # Preprocess for MiDaS small (256x256 input)
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (256, 256), (123.675, 116.28, 103.53), True, False)
        net.setInput(blob)
        
        # Forward pass
        depth = net.forward()
        depth = depth[0, :, :]
        
        # Resize back to original image size
        depth_resized = cv2.resize(depth, (w, h))
        
        # Normalize to 0-255
        depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        os.makedirs("scratch", exist_ok=True)
        cv2.imwrite("scratch/test_midas_raw_depth.png", depth_norm)
        print("Saved raw MiDaS depth to scratch/test_midas_raw_depth.png")

if __name__ == "__main__":
    test_midas_depth()
