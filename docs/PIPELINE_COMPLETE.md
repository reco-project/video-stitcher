# Backend Video Processing Pipeline

## Implementation Status

### Pipeline Components ✅

- [x] Video transcoding with FFmpeg (audio sync and vertical stacking)
- [x] Frontend frame extraction using Three.js shaders (FrameExtractor.jsx)
- [x] Feature matching with OpenCV SIFT/ORB
- [x] RANSAC outlier filtering
- [x] Position optimization with scipy
- [x] Match status tracking (pending → transcoding → awaiting_frames → ready)
- [x] Recalibration for ready matches
- [x] Error handling with structured codes
- [x] Debug visualization (feature matches image)

## Backend Endpoints

```bash
POST   /api/transcode                   # Stack videos (→ awaiting_frames)
POST   /api/process-with-frames         # Calibrate from warped frames
GET    /api/matches                     # List matches
POST   /api/matches                     # Create match
GET    /api/profiles                    # List profiles
POST   /api/profiles                    # Create profile
```

## Processing Pipeline

1. Audio synchronization via cross-correlation
2. Vertical video stacking with offset
3. Frontend frame extraction with Three.js (FrameExtractor.jsx)
4. SIFT/ORB feature matching
5. RANSAC outlier filtering
6. Position optimization (cameraAxisOffset, intersect, angles)
7. Match status updates with progress tracking
8. Error handling with structured codes
9. Debug visualization (feature_matches.png)

## Frontend Integration

- **FrameExtractor.jsx** - React Three Fiber component for frame warping
- **Recalibrate button** - Re-extract frames without transcoding
- **Status display** - Shows transcoding/feature_matching/optimizing
- **Error handling** - Toast notifications for failures
- **Progress UI** - Visual feedback during extraction

## Requirements

- **FFmpeg**: Must be in PATH for video transcoding
- **Python Dependencies**: Listed in `requirements.txt`
