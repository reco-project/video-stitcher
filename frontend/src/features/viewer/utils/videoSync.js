/**
 * Video synchronization utility using requestVideoFrameCallback API.
 *
 * Provides frame-accurate synchronization between two video elements
 * without requiring transcoding/stacking.
 */

/**
 * Check if requestVideoFrameCallback is supported
 * @returns {boolean}
 */
export function isRVFCSupported() {
	return 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
}

/**
 * Synchronize two video elements using requestVideoFrameCallback.
 *
 * @param {HTMLVideoElement} video1 - Primary video (typically left camera)
 * @param {HTMLVideoElement} video2 - Secondary video (typically right camera)
 * @param {number} offsetSeconds - Time offset between videos (video2 = video1 + offset)
 * @param {Function} onFrame - Callback fired on each frame with (video1Metadata, video2Metadata)
 * @returns {Function} Cleanup function to stop sync
 */
export function syncVideosWithRVFC(video1, video2, offsetSeconds, onFrame = null) {
	if (!isRVFCSupported()) {
		console.warn('requestVideoFrameCallback not supported, falling back to requestAnimationFrame');
		return syncVideosWithRAF(video1, video2, offsetSeconds, onFrame);
	}

	let callbackId1 = null;
	let callbackId2 = null;
	let isActive = true;

	// Metadata tracking
	let lastMetadata1 = null;
	let lastMetadata2 = null;

	const SYNC_THRESHOLD = 500; // 500ms tolerance for sync drift

	const handleFrame1 = (now, metadata) => {
		if (!isActive) return;

		lastMetadata1 = metadata;

		// Calculate target time for video2 based on video1's mediaTime
		const targetTime2 = metadata.mediaTime + offsetSeconds;

		// Check if video2 needs adjustment
		if (video2 && Math.abs(video2.currentTime - targetTime2) > SYNC_THRESHOLD) {
			console.log(`Sync drift detected: ${(video2.currentTime - targetTime2).toFixed(3)}s, correcting...`);
			video2.currentTime = targetTime2;
		}

		// Fire callback if both metadata available
		if (onFrame && lastMetadata2) {
			onFrame(lastMetadata1, lastMetadata2);
		}

		// Re-register for next frame
		if (isActive) {
			callbackId1 = video1.requestVideoFrameCallback(handleFrame1);
		}
	};

	const handleFrame2 = (now, metadata) => {
		if (!isActive) return;

		lastMetadata2 = metadata;

		// Fire callback if both metadata available
		if (onFrame && lastMetadata1) {
			onFrame(lastMetadata1, lastMetadata2);
		}

		// Re-register for next frame
		if (isActive) {
			callbackId2 = video2.requestVideoFrameCallback(handleFrame2);
		}
	};

	// Initial registration
	callbackId1 = video1.requestVideoFrameCallback(handleFrame1);
	callbackId2 = video2.requestVideoFrameCallback(handleFrame2);

	// Return cleanup function
	return () => {
		isActive = false;
		if (callbackId1 !== null) {
			video1.cancelVideoFrameCallback(callbackId1);
		}
		if (callbackId2 !== null) {
			video2.cancelVideoFrameCallback(callbackId2);
		}
	};
}

/**
 * Fallback sync using requestAnimationFrame for browsers without rVFC support.
 *
 * @param {HTMLVideoElement} video1 - Primary video
 * @param {HTMLVideoElement} video2 - Secondary video
 * @param {number} offsetSeconds - Time offset
 * @param {Function} onFrame - Callback fired on each frame
 * @returns {Function} Cleanup function
 */
function syncVideosWithRAF(video1, video2, offsetSeconds, onFrame = null) {
	let rafId = null;
	let isActive = true;

	const SYNC_THRESHOLD = 0.05;

	const updateLoop = () => {
		if (!isActive) return;

		// Sync video2 to video1 + offset
		const targetTime2 = video1.currentTime + offsetSeconds;
		if (video2 && Math.abs(video2.currentTime - targetTime2) > SYNC_THRESHOLD) {
			video2.currentTime = targetTime2;
		}

		// Fire callback with synthetic metadata
		if (onFrame) {
			onFrame(
				{ mediaTime: video1.currentTime, currentTime: video1.currentTime },
				{ mediaTime: video2.currentTime, currentTime: video2.currentTime }
			);
		}

		rafId = requestAnimationFrame(updateLoop);
	};

	rafId = requestAnimationFrame(updateLoop);

	return () => {
		isActive = false;
		if (rafId !== null) {
			cancelAnimationFrame(rafId);
		}
	};
}

/**
 * Initialize two videos and start them in sync with a given offset.
 *
 * @param {HTMLVideoElement} video1 - Primary video
 * @param {HTMLVideoElement} video2 - Secondary video
 * @param {number} offsetSeconds - Time offset between videos
 * @returns {Promise<Function>} Resolves with cleanup function when both videos ready
 */
export async function initializeSyncedVideos(video1, video2, offsetSeconds) {
	// Wait for both videos to be ready
	await Promise.all([
		new Promise((resolve) => {
			if (video1.readyState >= 2) {
				resolve();
			} else {
				video1.addEventListener('loadeddata', resolve, { once: true });
			}
		}),
		new Promise((resolve) => {
			if (video2.readyState >= 2) {
				resolve();
			} else {
				video2.addEventListener('loadeddata', resolve, { once: true });
			}
		}),
	]);

	// Seek video2 to initial offset position
	video2.currentTime = offsetSeconds;
	await new Promise((resolve) => {
		video2.addEventListener('seeked', resolve, { once: true });
	});

	// Start sync
	const cleanup = syncVideosWithRVFC(video1, video2, offsetSeconds);

	return cleanup;
}
