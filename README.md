# Hackathon Proposal: Advancing 3D Reconstruction and Stereoscopic Visualization

**Contact:** gogazago@google.com

## Project Description

This document outlines a project framework for developing an end-to-end pipeline capable of transforming 2D video into immersive 3D experiences. By combining state-of-the-art video-to-3D reconstruction with stereoscopic rendering techniques, we aim to demonstrate a scalable solution for volumetric media.

## Project Scope

| Phase | Objective | Key technology |
|-------|-----------|-----------------|
| Phase 1 | Video to 3D Rendering | Neural Radiance Fields (NeRF) or 3D Gaussian Splatting |
| Phase 2 | 3D Rendering to Stereoscopic View | Binocular disparity rendering (Left/Right eye projection) |

## Technical Roadmap

### 1. Video to 3D Rendering

- **Data Pre-processing:** Frame extraction and camera pose estimation using Structure-from-Motion (SfM).
- **Optimization:** Utilizing 3D Gaussian Splatting for high-fidelity scene representation.
- **Challenge:** Managing temporal consistency and occlusions within dynamic scenes.

### 2. Stereoscopic Integration for Google Cardboard

To enable viewing on Google Cardboard, the pipeline will implement the following:

- **Virtual Camera Setup:** Implementing dual virtual cameras with a 64mm Inter-Pupillary Distance (IPD) offset.
- **Viewport Configuration:** Rendering the Left Eye view to the left 50% of the display and the Right Eye view to the right 50%.
- **Distortion Correction:** Applying a "barrel distortion" shader to compensate for the spherical lenses in the Cardboard headset, ensuring a corrected, linear image.
- **Projection Strategy:** Utilizing parallel projection to minimize vertical parallax and reduce user eye strain.

## Evaluation Metrics

- **Visual Fidelity:** SSIM (Structural Similarity Index) and PSNR comparisons.
- **Rendering Latency:** Frames-per-second (FPS) output for real-time stereoscopic visualization.
- **Spatial Accuracy:** Precision of depth reconstruction in the final stereo pair.
