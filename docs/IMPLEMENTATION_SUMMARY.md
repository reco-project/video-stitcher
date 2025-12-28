# Video Processing Pipeline - Implementation Summary

## Overview

Complete backend video processing pipeline for automatic calibration and stitching preparation.

## Architecture

### Pipeline Flow

```
1. User creates match → assigns videos + lens profiles
2. POST /api/transcode → audio sync + vertical stacking (status: awaiting_frames)
3. Frontend extracts frames → FrameExtractor.jsx warps using Three.js shaders
4. POST /api/process-with-frames → receives warped frames
5. Feature matching → SIFT/ORB detection + RANSAC filtering
6. Position optimization → scipy minimize angles
7. Match updated → status=ready, src=stacked_video, params=calibrated
8. Recalibration → skip step 2, re-extract frames, re-optimize
```

### Components

#### Frontend

- **`FrameExtractor.jsx`**: React Three Fiber frame extraction
    - Fixed 1920x1080 render resolution
    - Uses viewer geometry and camera FOV
    - Sequential left→right extraction
    - Captures canvas to PNG blobs

#### Backend Services

- **`transcoding.py`**: FFmpeg audio sync and stacking
- **`feature_matching.py`**: OpenCV SIFT/ORB feature detection with RANSAC
- **`position_optimization.py`**: Scipy optimization for camera position

#### API Endpoints

- **`POST /api/transcode`**: Stack videos → awaiting_frames
- **`POST /api/process-with-frames`**: Calibrate from warped frames → ready

### Storage Structure

```
backend/
  data/
    matches/          # Match metadata JSON
    lens_profiles/    # Calibration data
  temp/               # Temporary processing files
    {match_id}/
      stacked_video.mp4
      debug_frames/
        left_received.png
        right_received.png
        feature_matches.png
```
