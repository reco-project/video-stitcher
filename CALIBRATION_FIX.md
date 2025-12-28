# Calibration Failure Fix: Blank Frame Extraction

## Root Cause Analysis

The calibration was failing because **blank/black frames were being extracted** from the video. This resulted in the backend's feature matching failing with: `"Images appear to be blank (std: left=X.XX, right=X.XX)"`.

### The Problem Chain

1. **Race Condition in Frame Extraction** (`FrameExtractor.jsx`):
    - The `ExtractionCanvas` was attempting to render WebGL before the video frame was actually loaded
    - `useCustomVideoTexture` was called with `null` source, so it relied on an external video element
    - The texture was created and Canvas rendered before `videoElement.currentTime` was properly set with actual frame data

2. **Premature readyState Check**:
    - `handleSeeked` was marking video as ready (`setVideoReady(true)`) immediately upon seeking
    - However, `readyState >= 2` only means metadata is loaded, NOT that pixel data is available
    - The shader would render before the video frame texture was populated, resulting in black pixels

3. **Insufficient Wait Conditions**:
    - No validation that `videoElement.currentTime` matched the target `frameTime`
    - No check for `readyState >= 3` (HAVE_FUTURE_DATA/HAVE_ENOUGH_DATA) before rendering
    - Video element dimensions could still be 0 during rendering

## Solution Implemented

### 1. **ExtractionCanvas - Add Frame Readiness Check** (lines 73-108)

```javascript
// Wait for video frame to actually be available at current time before rendering
useEffect(() => {
	const checkFrameReady = () => {
		// Only ready when ALL conditions are met:
		if (
			videoElement.readyState >= 3 && // HAVE_FUTURE_DATA or HAVE_ENOUGH_DATA
			videoElement.currentTime > 0 &&
			videoElement.videoWidth > 0 &&
			videoElement.videoHeight > 0
		) {
			setVideoFrameReady(true);
		}
	};

	videoElement.addEventListener('canplay', checkFrameReady);
	videoElement.addEventListener('seeked', checkFrameReady);
	checkFrameReady(); // Check immediately
}, [videoElement]);
```

### 2. **ExtractionCanvas - Conditional Rendering** (lines 130-148)

```javascript
{
	texture && videoFrameReady ? <Canvas>...</Canvas> : <div>Loading frame...</div>;
}
```

Canvas only renders when both texture AND frame are actually ready.

### 3. **FrameExtractor - Add canplay Event Handler** (lines 193-199)

```javascript
const handleCanplay = () => {
	console.log('Video canplay event');
	// Can now access pixel data, trigger seek if needed
	if (video.currentTime !== frameTime) {
		video.currentTime = frameTime;
	}
};
video.addEventListener('canplay', handleCanplay);
```

Ensure seeking happens at the right time in the video lifecycle.

### 4. **FrameExtractor - Proper readyState Validation** (lines 180-189)

```javascript
// CRITICAL FIX: Only mark as ready once we have actual frame data
if (video.readyState >= 3 && video.videoWidth > 0 && video.videoHeight > 0) {
	setVideoReady(true);
} else {
	console.warn('Video not ready yet...', { readyState, videoWidth, videoHeight });
}
```

Only signal readiness when pixel data is truly available.

### 5. **Error Handling for Empty Frames** (lines 239-244, 253-258)

```javascript
if (!blob || blob.size === 0) {
	console.error('Left frame is empty!');
	onError?.(new Error('Left frame extraction resulted in empty image'));
	return;
}
```

Catch and report empty frames immediately instead of sending blank data to backend.

## Video State Lifecycle

```
loadstart → loadedmetadata → canplay → seeked → (frame ready)
```

**Before Fix:**

- Stream would mark as ready on `seeked` event alone
- Canvas would render with uninitialized texture
- Result: Black/blank PNG sent to backend

**After Fix:**

- Stream checks `readyState >= 3` + valid `currentTime` + valid dimensions
- Canvas only renders when `videoFrameReady === true`
- Ensures WebGL shader has valid pixel data to process
- Result: Valid warped frame PNG sent to backend

## Testing

To verify the fix works:

1. Extract a frame from a match in the UI
2. Check browser console for:
    - `"Video seeked to: X"` (target frame time)
    - `"Video canplay event"` (frame ready)
    - `"Capturing canvas"` (extraction starting)
    - `"Blob created: XXXX bytes"` (frame successfully captured)

3. Verify backend debug frames:
    - `temp/{match_id}/debug_frames/left_received.png` - should show warped video frame
    - `temp/{match_id}/debug_frames/right_received.png` - should show warped video frame
    - `temp/{match_id}/debug_frames/feature_matches.png` - should show detected features

4. Check calibration status:
    - Match should progress to `"calibrating"` → `"ready"`
    - Stored params should be valid camera calibration results

## Impact

- **Fixes blank frame extraction** - frames will now contain valid warped video pixels
- **Enables feature matching** - backend will have valid pixel data for SIFT/ORB detection
- **Allows calibration to succeed** - sufficient features will be found to optimize camera positions
- **Better error reporting** - empty frames are caught immediately with clear error messages

## Files Changed

- `frontend/src/features/viewer/components/FrameExtractor.jsx` - Frame extraction timing fixes
