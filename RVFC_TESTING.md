# Testing the rVFC Prototype

## Quick Test Setup

Since the backend currently generates stacked videos, here's how to test the prototype with minimal changes:

### Option 1: Use Existing Separate Videos (Recommended)

If you still have the original left/right video files:

1. **Update a match JSON manually** (e.g., `backend/data/matches/m1.json`):

```json
{
  "id": "m1",
  "name": "Test Match",
  "left_video_path": "videos/left-camera.mp4",
  "right_video_path": "videos/right-camera.mp4",
  "audio_offset": 0.0,
  "params": { ... },
  "left_uniforms": { ... },
  "right_uniforms": { ... }
}
```

2. **Place videos in `backend/data/videos/`**:
    - `left-camera.mp4`
    - `right-camera.mp4`

3. **Start dev server** and toggle on "Dual Video Mode"

### Option 2: Simulate with Test Videos

Create two simple test videos with `ffmpeg`:

```bash
# Generate test video 1 (left camera)
ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 \
  -f lavfi -i sine=frequency=440:duration=10 \
  backend/data/videos/test-left.mp4

# Generate test video 2 (right camera) with 0.5s offset
ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 \
  -f lavfi -i sine=frequency=880:duration=10 \
  backend/data/videos/test-right.mp4
```

Then create a test match JSON with these paths and `audio_offset: 0.5`.

### Option 3: Split Existing Stacked Video

If you only have stacked videos:

```bash
# Extract top half (left camera)
ffmpeg -i stacked-video.mp4 -vf "crop=iw:ih/2:0:0" left.mp4

# Extract bottom half (right camera)
ffmpeg -i stacked-video.mp4 -vf "crop=iw:ih/2:0:ih/2" right.mp4
```

## Testing Checklist

Once you have two video files:

- [ ] Videos load without errors
- [ ] Both video canvases display frames
- [ ] Play button works
- [ ] Videos stay in sync during playback
- [ ] Seeking maintains sync
- [ ] Console shows sync corrections (if drift detected)
- [ ] No significant performance degradation

## What to Look For

### Good Signs ✅

- Smooth playback
- Minimal sync corrections in console
- Both videos advance together
- No frame drops

### Bad Signs ❌

- Constant drift corrections
- One video freezes
- Audio/video desync
- High CPU usage
- Frame drops

## Browser Testing

Test in multiple browsers:

```bash
# Check rVFC support in console:
console.log('requestVideoFrameCallback' in HTMLVideoElement.prototype);
```

- **Chrome/Edge**: Should use native rVFC
- **Safari 15.4+**: Should use native rVFC
- **Firefox**: Will fall back to RAF (expected)

## Debugging

Enable verbose logging in `videoSync.js`:

```javascript
// Add at the top of syncVideosWithRVFC:
const DEBUG = true;

if (DEBUG) {
	console.log('Sync frame:', metadata.mediaTime, 'Drift:', drift);
}
```

## Performance Comparison

Compare with stacked video mode:

1. **Load time**: How fast do videos start?
2. **Seek time**: How responsive is seeking?
3. **CPU usage**: Check browser task manager
4. **Memory usage**: Monitor over time

## Next Steps After Testing

If sync works well:

- [ ] Update backend to skip transcoding
- [ ] Integrate into main 3D viewer
- [ ] Update frame extraction logic
- [ ] Add sync quality metrics
- [ ] Test with production videos

If sync has issues:

- [ ] Document specific problems
- [ ] Test with different video formats
- [ ] Adjust sync threshold
- [ ] Consider hybrid approach (stacked fallback)
