# GPU Encoder Configuration

## Overview

The application now supports configurable GPU encoder selection for FFmpeg video transcoding. Users can choose between automatic detection and specific hardware encoders (NVIDIA, Intel, AMD) or CPU software encoding.

## Features

### Backend Settings (`backend/app/config.py`)
- **Persistent Settings**: Encoder preference is saved to `backend/data/settings.json`
- **Supported Encoders**:
  - `auto` - Automatically detect best available (NVIDIA > Intel > AMD > CPU)
  - `h264_nvenc` - NVIDIA GPU encoder
  - `h264_qsv` - Intel Quick Sync Video
  - `h264_amf` - AMD Advanced Media Framework
  - `libx264` - CPU software encoding

### API Endpoints (`backend/app/routers/settings.py`)
- `GET /api/settings/encoders` - Get available encoders and current preference
- `PUT /api/settings/encoders` - Update encoder preference
- `GET /api/settings/` - Get all application settings

### Transcoding Service Updates (`backend/app/services/transcoding.py`)
- Respects user encoder preference from settings
- Falls back gracefully if preferred encoder unavailable
- Logs encoder selection for debugging

### Frontend UI (`frontend/src/features/settings/components/AppSettings.jsx`)
- GPU Acceleration card in Settings page
- Dropdown to select encoder preference
- Displays available hardware encoders
- Real-time updates without restart

## Usage

### For Users
1. Navigate to Settings page in the app
2. Find "GPU Acceleration" section at the top
3. Select your preferred encoder from dropdown:
   - **Auto-detect** (Recommended) - Automatically uses best available
   - **NVIDIA GPU** - Force NVIDIA hardware encoding
   - **Intel GPU** - Force Intel Quick Sync
   - **AMD GPU** - Force AMD hardware encoding
   - **CPU (libx264)** - Software encoding (slowest, most compatible)
4. Setting takes effect immediately for new transcoding jobs

### For Developers

#### Check Available Encoders
```bash
ffmpeg -hide_banner -encoders | grep h264
```

#### Test Encoder Detection
```bash
curl http://127.0.0.1:8000/api/settings/encoders | jq
```

#### Update Encoder Preference
```bash
curl -X PUT http://127.0.0.1:8000/api/settings/encoders \
  -H "Content-Type: application/json" \
  -d '{"encoder": "h264_nvenc"}'
```

## Troubleshooting

### Friend's 4070 Ti Not Being Used
**Problem**: User's friend has NVIDIA 4070 Ti but CPU encoding was being used.

**Solution**: 
1. Check if NVIDIA encoder is available:
   ```bash
   ffmpeg -hide_banner -encoders | grep nvenc
   ```
2. If available, select "NVIDIA GPU (h264_nvenc)" in Settings
3. If not available:
   - Install/update NVIDIA drivers
   - Install ffmpeg with NVENC support: `ffmpeg -version | grep nvenc`
   - On Linux: May need ffmpeg compiled with `--enable-nvenc`
   - On Windows: Download ffmpeg build with NVENC support

### Encoder Not Available
If your preferred encoder isn't in the dropdown:
- The hardware isn't detected by FFmpeg
- FFmpeg needs to be recompiled with hardware encoding support
- Drivers need to be installed/updated

### Performance Comparison
Approximate encoding speeds (1080p @ 30fps):
- **NVIDIA GPU (h264_nvenc)**: 300-600 FPS (10-20x realtime)
- **Intel QSV (h264_qsv)**: 100-300 FPS (3-10x realtime)
- **AMD AMF (h264_amf)**: 100-300 FPS (3-10x realtime)
- **CPU (libx264)**: 20-60 FPS (1-2x realtime)

Actual performance depends on:
- GPU model and generation
- Video resolution and complexity
- CPU if using software encoding
- Available VRAM

## Technical Details

### Settings Persistence
Settings are stored in JSON format at: `backend/data/settings.json`

Example:
```json
{
  "encoder": "h264_nvenc"
}
```

### Encoder Detection Logic
1. Load user preference from settings
2. If preference is not "auto", try to use it
3. If unavailable or "auto", detect in priority order:
   - NVIDIA (h264_nvenc)
   - Intel (h264_qsv)
   - AMD (h264_amf)
   - CPU fallback (libx264)
4. Log selected encoder for debugging

### Frontend Architecture
- Settings API client: `frontend/src/features/settings/api/settings.js`
- UI Component: `frontend/src/features/settings/components/AppSettings.jsx`
- Loads available encoders on mount
- Updates immediately via PUT request
- No app restart required

## Future Enhancements

Potential improvements:
- [ ] Add encoder quality presets (fast, balanced, quality)
- [ ] Show estimated encoding speed for each encoder
- [ ] Add H.265/HEVC encoder support
- [ ] Profile-specific encoder selection
- [ ] Benchmark tool to compare encoder performance
- [ ] GPU memory usage monitoring during encoding
