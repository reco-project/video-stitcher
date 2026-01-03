# rVFC Prototype (feat/rvfc branch)

This branch contains a minimal prototype implementation of `requestVideoFrameCallback()` for dual-video synchronization.

## Goal

Test whether we can eliminate the video transcoding/stacking step by:

1. Loading two separate video files (left + right camera)
2. Using `requestVideoFrameCallback()` for frame-accurate synchronization
3. Rendering both videos to separate textures in the viewer

## What's Been Added

### Core Sync Module

**`frontend/src/features/viewer/utils/videoSync.js`**

- `isRVFCSupported()` - Feature detection
- `syncVideosWithRVFC()` - Main sync function using rVFC
- `syncVideosWithRAF()` - Fallback for unsupported browsers
- `initializeSyncedVideos()` - High-level initialization

### Dual Video Texture Hook

**`frontend/src/features/viewer/hooks/useDualVideoTextures.js`**

- React hook that manages two video elements + textures
- Handles sync initialization and cleanup
- Returns both textures for use in Three.js scene

### Test Component

**`frontend/src/features/viewer/components/DualVideoTest.jsx`**

- Minimal UI to test dual-video sync
- Side-by-side canvas rendering
- Play/pause controls

### Viewer Integration

**`frontend/src/features/viewer/components/Viewer.jsx`**

- Added prototype toggle switch in settings panel
- Allows switching between stacked video (current) and dual-video (prototype)

## How to Test

1. **Start the dev environment:**

    ```bash
    npm run dev
    ```

2. **Expand the viewer settings panel** (click the header to expand)

3. **Enable "Dual Video Mode (rVFC Prototype)"** toggle

4. **Current Limitation:** The backend currently only provides stacked video paths. To properly test:
    - Backend needs to return `left_video_path` and `right_video_path` in match data
    - Or manually place two separate video files and update match JSON

## Browser Support

| Browser      | rVFC Support | Fallback       |
| ------------ | ------------ | -------------- |
| Chrome 83+   | ✅ Native    | -              |
| Edge 83+     | ✅ Native    | -              |
| Safari 15.4+ | ✅ Native    | -              |
| Firefox      | ❌           | RAF-based sync |

## Technical Details

### Synchronization Strategy

1. **Primary video** (left camera) drives the sync
2. On each frame callback, check secondary video's `currentTime`
3. If drift > 50ms threshold, seek secondary to match
4. Use `metadata.mediaTime` for accurate frame identification (not `video.currentTime`)

### Key Metadata Properties

From `VideoFrameMetadata`:

- `mediaTime` - Presentation timestamp (PTS) of the frame
- `presentedFrames` - Count of submitted frames (detect skips)
- `expectedDisplayTime` - When frame will be visible
- `captureTime` / `receiveTime` - For WebRTC sources

### Advantages Over Current Approach

✅ **No transcoding delay** - Videos load immediately  
✅ **No disk space waste** - No stacked video files  
✅ **Frame-accurate sync** - Using actual video PTS  
✅ **Simpler pipeline** - Backend only computes offset  
✅ **Better architecture** - Separation of concerns

### Remaining Work (if prototype succeeds)

- [ ] Update backend to skip video stacking
- [ ] Backend returns audio offset + separate video paths
- [ ] Integrate dual textures into main 3D viewer
- [ ] Update `VideoPlane` component to use separate textures
- [ ] Handle edge cases (seeking, playback rate changes)
- [ ] Add sync quality metrics/debugging UI
- [ ] Update frame extraction for calibration
- [ ] Test with various video formats and frame rates

## Next Steps

1. **Validate sync quality** - Does it stay in sync during playback?
2. **Measure performance** - Any CPU/GPU overhead vs stacked video?
3. **Test edge cases** - Seeking, pausing, playback speed
4. **Decide on integration** - If successful, plan full integration

## Rollback Plan

This is a feature branch. If the prototype doesn't work well:

- Simply don't merge to main
- No changes to production code
- Easy to abandon or revisit later

## References

- [requestVideoFrameCallback() Documentation](https://web.dev/articles/requestvideoframecallback-rvfc)
- [WebCodecs API](https://developer.chrome.com/docs/web-platform/best-practices/webcodecs)
- [Original Discussion in README](../README.md) (search for "rVFC demo")
