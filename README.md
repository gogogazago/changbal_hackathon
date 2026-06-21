# 🎧 Hackathon: 3D Reconstruction and Stereoscopic Visualization for Google Cardboard

**Contact:** gogazago@google.com

---

## 🌟 Project Description

This project implements an end-to-end pipeline that transforms a **2D video of a real-world object** (e.g., headphones, bananas) into immersive **3D reconstructions** and **stereoscopic visualizations**. By combining frame extraction, pixel-by-pixel analysis, depth estimation, 3D point cloud generation, and OBJ mesh export, we demonstrate a scalable approach to volumetric media creation — suitable for viewing on mobile devices and **Google Cardboard VR**.

### Hero Preview: 3D Gaussian Splatting Reconstruction

![3D Gaussian Splatting Reconstruction](3d_reconstruction_hero.png)

---

## 📂 Project Structure

```
changbal_hackathon/
├── README.md                       # This file
├── LICENSE                         # Project license
├── 3d_reconstruction_hero.png      # Hero preview visualization
├── extract_frames.py               # Main pipeline script (auto-discovers objects & exports OBJ)
├── create_3d_video.py              # 3D floating rotating video generator
├── model-small.onnx                # MiDaS depth estimation ONNX model
├── video/                          # ← Source videos organized by object folder
│   ├── banana/
│   │   └── PXL_20260620_172028913.mp4   # Source video for banana
│   └── headphone/
│       └── video_headphone.mp4          # Source video for headphone
└── pixel_frames/                   # ← Generated outputs per object
    ├── banana/
    │   ├── frames/                 # Extracted key frames
    │   ├── pixel_grid/             # Pixel-by-pixel grid visualizations
    │   ├── depth_maps/             # Inferno depth estimations
    │   └── 3d_renders/             # 3D point cloud SBS VR renders & OBJ export
    └── headphone/
        ├── frames/                 # Extracted key frames (24 frames)
        ├── pixel_grid/             # Pixel-by-pixel grid visualizations
        ├── depth_maps/             # Inferno depth estimations
        └── 3d_renders/             # 3D point cloud SBS VR renders
            ├── frame_0000_3d.png   # Stereoscopic Side-by-Side (SBS) render
            ├── headphone.obj       # 3D mesh (Vertices: 28528, Faces: 56640)
            ├── headphone.mtl       # Material definition file
            └── headphone_texture.png # Generated texture atlas (unwrapped UV space)
```

---

## 🎬 Pipeline Overview (`extract_frames.py`)

The pipeline consists of **5 automated stages** that discover object folders under `video/` and generate outputs under `pixel_frames/<object>/`:

### Stage 1: Frame Extraction
Extracts key frames at regular intervals (every 10th frame) from the source video in `video/<object>/`.
* **Resolution:** 1080 × 1920 (portrait)
* **Frame Rate:** 30.0 fps
* **Headphone Frames:** 24 key frames extracted.

### Stage 2: Pixel-by-Pixel Grid Visualization
Each key frame is downscaled to a **64×64 pixel grid** and enlarged using nearest-neighbor interpolation. This maps raw pixel color blocks to prepare them for coordinate translation.

![Pixel Grid Example](pixel_frames/headphone/pixel_grid/frame_0000_pixels.png)

### Stage 3: Depth Map Estimation
Uses the **MiDaS ONNX model** (or fallback Sobel filters) to estimate relative depth values. The output is normalized within the object mask and visualized with the **Inferno colormap** (brighter colors indicate closer depth).

![Depth Map Example](pixel_frames/headphone/depth_maps/frame_0000_depth.png)

### Stage 4: 3D Point Cloud SBS VR Rendering (Google Cardboard)
Projects isolated foreground pixels into **3D space** using a dual-perspective camera setup to generate a **stereoscopic Side-by-Side (SBS)** pair:
* **Yaw Parallax (양안 시점):** Rendered with a Left eye (-0.02 rad) and Right eye (+0.02 rad) offset.
* **Centroid Stabilization:** Tracks the object's center of mass in 3D to align it perfectly at the center of each eye channel to prevent drift and eye strain.
* **Feathered Splatting:** Smooths point sizes based on distance boundaries.

![3D Point Cloud](pixel_frames/headphone/3d_renders/frame_0000_3d.png)

### Stage 5: Exporting OBJ 3D Mesh & Texture Atlas
Constructs a **layered 3D surface mesh** by connecting neighboring foreground pixels across multiple depth layers.
* **Unwrapped UV Atlas:** Generates a clean texture atlas (`headphone_texture.png`) containing both front-facing and back-facing texture mappings.
* **OBJ Mesh Output:** Exports standard `.obj` and `.mtl` files.
* **Pre-rotation Alignment:** The mesh is pre-rotated (Y-axis: 178°, X-axis: 21° for headphones) to render in a perfectly upright, facing orientation in mobile Cardboard viewer engines.

---

## 🎥 Floating 3D Rotating Video Showcase (`create_3d_video.py`)

Generates a continuous 360-degree rotating video showcase of the segmented object:
* **Background Removal:** Uses 3D plane fitting on estimated relative depth maps combined with selective color HSV masks to remove desks and office backgrounds.
* **Stabilized Rotation:** Renders the isolated foreground object spinning in the center of the frame without vertical/horizontal sway.
* **Output Path:** `pixel_frames/<object>/<object>_3d_reconstruction.mp4`

---

## 🏗 Project Scope & Roadmap

| Phase | Objective | Status / Key Technology |
|-------|-----------|-------------------------|
| **Phase 1** | Video to 3D OBJ Mesh | **Completed** (Multi-layer depth mesh export) |
| **Phase 2** | Mobile stereoscopic viewer integration | **Completed** (Dual-perspective SBS images & OBJ pre-rotation) |
| **Phase 3** | Android / iOS native rendering | Android custom renderer (`java_src/`) |

---

## 🚀 How to Run

### Prerequisites

Install the required Python libraries:
```bash
pip3 install opencv-python Pillow numpy
```

### Run the Frame Extraction and 3D Export Pipeline

```bash
# Runs the full 5-stage pipeline for all discovered videos
python3 extract_frames.py
```

### Run the Rotating 3D Video Showcase

```bash
# Generates a rotating MP4 video showcase for a specific object
python3 create_3d_video.py headphone
python3 create_3d_video.py banana
```

### Adding a New Object

To process a new custom video, create a directory under `video/` and copy the video file:
```bash
mkdir video/my_new_object
cp my_video.mp4 video/my_new_object/
python3 extract_frames.py
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
