# Live Streaming Guide

This guide explains how to use the Live Match feature to view and process live video streams in real-time.

## Overview

The Live Match is a special match that automatically appears at the top of your match list. It supports:
- Real-time video streaming (HLS format)
- Lens profile selection and switching
- Live recalibration and color correction
- All standard viewer features (zoom, rotation, etc.)

**Important:** Your source video must be two fisheye views stacked vertically (top and bottom). The app will split and process each half as left and right views for 360° playback.

## Setup Options

### Option 1: UDP Stream from Raspberry Pi

Stream video from your Pi to the PC via UDP, which the backend converts to HLS.

**On Raspberry Pi:**
```bash
# Stream video to PC at 192.168.1.100 via UDP
ffmpeg -f v4l2 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency \
  -b:v 2M -f mpegts udp://192.168.1.100:5000
```

**On PC:**
1. Start the Live stream via the backend API:
   ```bash
   curl -X POST http://127.0.0.1:8000/live/start
   ```
2. The backend will ingest UDP on port 5000 and serve as HLS
3. Open the Live match in the app - it will use `videos/live/index.m3u8`

**Stop streaming:**
```bash
curl -X POST http://127.0.0.1:8000/live/stop
```

### Option 2: External HLS URL

Use any existing HLS stream (live TV, remote camera, etc.).

1. Edit `devData/matches/live.json`
2. Set the `src` field to your HLS URL:
   ```json
   {
     "id": "live",
     "src": "http://example.com/stream/index.m3u8",
     ...
   }
   ```
3. Refresh the app - the Live match will use your custom URL

**Note:** The app only sets `src` if it's empty. Your custom URLs are preserved across restarts.

## Using the Live Match

1. **Select the Live match** from the top of your match list
2. **Choose a lens profile** from the Live Profiles panel on the right
3. **Adjust settings:**
   - Recalibration parameters (horizon line, keystone, etc.)
   - Color correction and matching
   - Zoom, rotation, and all standard viewer controls
4. **Switch profiles on the fly** to compare different calibrations

## Configuration

### Live Stream Settings

The backend FFmpeg command generates:
- 2-second segments (low latency)
- H.264 baseline profile (maximum compatibility)
- 5-segment rolling window (auto-cleanup)

### HLS Player Settings

The player is tuned for live streaming:
- 15-second buffer for stability
- 3-segment sync target
- Automatic error recovery

## Troubleshooting

### "404 Not Found" for segments

The playlist references segments that don't exist. **Solution:**
1. Stop the stream: `curl -X POST http://127.0.0.1:8000/live/stop`
2. Clear old segments: `rm -f devData/videos/live/*`
3. Start fresh: `curl -X POST http://127.0.0.1:8000/live/start`
4. Refresh the app

### "bufferAppendError" spam

**Causes:**
- Codec incompatibility between segments
- Network issues causing duplicate/corrupt segments
- FFmpeg producing inconsistent encoding

**Solutions:**
1. Restart the stream (see above)
2. For external streams: Use a high-quality HLS source
3. Check network stability between Pi and PC

### Video freezes or stutters

1. **Check network:** Ensure stable connection between Pi and PC
2. **Reduce bitrate on Pi:** Lower the `-b:v` value in ffmpeg command
3. **Check CPU usage:** Re-encoding is CPU-intensive; reduce preset to `ultrafast`

### Stream won't start

1. **Check port availability:** `sudo lsof -i :5000`
2. **Kill existing process:** `pkill -f "ffmpeg.*udp://0.0.0.0:5000"`
3. **Check backend logs** for FFmpeg errors

## Technical Details

### File Locations

- **Live match data:** `devData/matches/live.json`
- **HLS segments:** `devData/videos/live/`
- **Backend endpoints:**
  - `POST /live/start` - Start UDP ingest
  - `POST /live/stop` - Stop ingest
  - `GET /live/status` - Check status
  - `GET /videos/live/index.m3u8` - Playlist
  - `GET /videos/live/seg_*.ts` - Segments

### No-Cache Headers

Live HLS content is served with `Cache-Control: no-cache` to prevent stale segment issues.

### Segment Management

With `delete_segments` flag enabled, only the last 5 segments are kept on disk, minimizing storage use.

## Tips

- **Use external HLS for production:** More stable than UDP→HLS conversion
- **Test latency:** Expect 6-10 seconds of delay (normal for HLS)
- **Save calibrations:** Create user profiles for your favorite live settings
- **Monitor logs:** Watch backend output for FFmpeg warnings
