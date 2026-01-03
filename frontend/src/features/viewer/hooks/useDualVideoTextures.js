import { useEffect, useState, useRef } from 'react';
import * as THREE from 'three';
import { useViewerStore } from '../stores/store.js';
import { env } from '@/config/env';
import { initializeSyncedVideos, isRVFCSupported } from '../utils/videoSync.js';

/**
 * Convert relative video path to absolute backend URL
 */
const resolveVideoUrl = (src) => {
	if (!src) return src;
	if (src.startsWith('http://') || src.startsWith('https://')) {
		return src;
	}
	const baseUrl = env.API_BASE_URL.replace('/api', '');
	return `${baseUrl}/${src}`;
};

/**
 * Hook for managing two separate synchronized video textures.
 *
 * Uses requestVideoFrameCallback for frame-accurate synchronization
 * without requiring video transcoding/stacking.
 *
 * @param {string} leftVideoSrc - Path to left camera video
 * @param {string} rightVideoSrc - Path to right camera video
 * @param {number} offsetSeconds - Time offset (right = left + offset)
 * @param {Object} options - Configuration options
 * @param {boolean} options.autoPlay - Auto-play videos (default: true)
 * @returns {Object} { leftTexture, rightTexture, videoElement: leftVideo, isReady }
 */
export const useDualVideoTextures = (leftVideoSrc, rightVideoSrc, offsetSeconds = 0, options = {}) => {
	const { autoPlay = true } = options;
	const setVideoRef = useViewerStore((s) => s.setVideoRef);
	const clearVideoRef = useViewerStore((s) => s.clearVideoRef);

	const [leftTexture, setLeftTexture] = useState(null);
	const [rightTexture, setRightTexture] = useState(null);
	const [isReady, setIsReady] = useState(false);

	const leftVideoRef = useRef(null);
	const rightVideoRef = useRef(null);
	const syncCleanupRef = useRef(null);

	useEffect(() => {
		if (!leftVideoSrc || !rightVideoSrc) return;

		// Check browser support
		if (!isRVFCSupported()) {
			console.warn('requestVideoFrameCallback not supported. Sync quality may be degraded.');
		}

		const leftVideoUrl = resolveVideoUrl(leftVideoSrc);
		const rightVideoUrl = resolveVideoUrl(rightVideoSrc);

		// Create video elements
		const leftVideo = document.createElement('video');
		leftVideo.crossOrigin = 'anonymous';
		leftVideo.preload = 'auto';
		leftVideo.playsInline = true;
		leftVideo.muted = false; // Left video has audio

		const rightVideo = document.createElement('video');
		rightVideo.crossOrigin = 'anonymous';
		rightVideo.preload = 'auto';
		rightVideo.playsInline = true;
		rightVideo.muted = true; // Right video muted to avoid echo

		leftVideoRef.current = leftVideo;
		rightVideoRef.current = rightVideo;

		// Create textures
		const leftVideoTexture = new THREE.VideoTexture(leftVideo);
		leftVideoTexture.minFilter = THREE.LinearFilter;
		leftVideoTexture.magFilter = THREE.LinearFilter;
		leftVideoTexture.generateMipmaps = false;

		const rightVideoTexture = new THREE.VideoTexture(rightVideo);
		rightVideoTexture.minFilter = THREE.LinearFilter;
		rightVideoTexture.magFilter = THREE.LinearFilter;
		rightVideoTexture.generateMipmaps = false;

		setLeftTexture(leftVideoTexture);
		setRightTexture(rightVideoTexture);

		// Store left video as primary control reference
		setVideoRef(leftVideo);

		// Load videos
		leftVideo.src = leftVideoUrl;
		rightVideo.src = rightVideoUrl;

		const initialize = async () => {
			try {
				// Wait for both videos to load
				await Promise.all([
					new Promise((resolve, reject) => {
						leftVideo.addEventListener('loadeddata', resolve, { once: true });
						leftVideo.addEventListener('error', reject, { once: true });
					}),
					new Promise((resolve, reject) => {
						rightVideo.addEventListener('loadeddata', resolve, { once: true });
						rightVideo.addEventListener('error', reject, { once: true });
					}),
				]);

				console.log('Both videos loaded, initializing sync...');
				console.log(`Offset: ${offsetSeconds}s`);

				// Initialize synchronization
				const cleanup = await initializeSyncedVideos(leftVideo, rightVideo, offsetSeconds);
				syncCleanupRef.current = cleanup;

				// Textures will be updated by the sync loop automatically

				// Start playing if autoPlay
				if (autoPlay) {
					await leftVideo.play();
					await rightVideo.play();
				}

				setIsReady(true);
				console.log('Dual video sync initialized successfully');
			} catch (error) {
				console.error('Failed to initialize dual video sync:', error);
			}
		};

		initialize();

		// Cleanup
		return () => {
			// Stop sync
			if (syncCleanupRef.current) {
				syncCleanupRef.current();
			}

			// Clean up videos
			if (leftVideo) {
				leftVideo.pause();
				leftVideo.removeAttribute('src');
				leftVideo.load();
			}
			if (rightVideo) {
				rightVideo.pause();
				rightVideo.removeAttribute('src');
				rightVideo.load();
			}

			// Clean up textures
			leftVideoTexture.dispose();
			rightVideoTexture.dispose();

			clearVideoRef();
			setIsReady(false);
		};
	}, [leftVideoSrc, rightVideoSrc, offsetSeconds, autoPlay, setVideoRef, clearVideoRef]);

	return {
		leftTexture,
		rightTexture,
		videoElement: leftVideoRef.current, // Return left as primary control
		isReady,
	};
};
