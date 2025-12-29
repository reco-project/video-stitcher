/**
 * Frame extraction and warping utility for calibration.
 *
 * Extracts frames from video, applies fisheye correction using Three.js shader,
 * and captures the warped result as image data for backend processing.
 */

import * as THREE from 'three';
import fisheyeShader from '../shaders/fisheye.js';
import { formatUniforms } from './utils.js';

/**
 * Extract and warp a frame from the video for calibration.
 *
 * @param {HTMLVideoElement} videoElement - The video element to extract from
 * @param {number} frameTime - Time in seconds to seek to
 * @param {Object} leftUniforms - Left camera lens uniforms
 * @param {Object} rightUniforms - Right camera lens uniforms
 * @returns {Promise<{leftBlob: Blob, rightBlob: Blob}>} Warped frame blobs
 */
export async function extractWarpedFrames(videoElement, frameTime, leftUniforms, rightUniforms) {
	return new Promise((resolve, reject) => {
		// Seek to the desired frame
		videoElement.currentTime = frameTime;

		videoElement.addEventListener(
			'seeked',
			async function onSeeked() {
				videoElement.removeEventListener('seeked', onSeeked);

				try {
					// Wait a bit for video to stabilize
					await new Promise((r) => setTimeout(r, 100));

					// Create visible container for debugging
					const container = document.createElement('div');
					container.style.position = 'fixed';
					container.style.top = '50%';
					container.style.left = '50%';
					container.style.transform = 'translate(-50%, -50%)';
					container.style.zIndex = '10000';
					container.style.background = 'rgba(0, 0, 0, 0.9)';
					container.style.padding = '20px';
					container.style.borderRadius = '8px';
					container.style.border = '2px solid #059669';

					const title = document.createElement('div');
					title.textContent = 'Extracting and warping frames...';
					title.style.color = 'white';
					title.style.marginBottom = '10px';
					title.style.fontWeight = 'bold';
					container.appendChild(title);

					document.body.appendChild(container);

					// Create Three.js scene - EXACTLY like the viewer
					const scene = new THREE.Scene();
					const renderer = new THREE.WebGLRenderer({
						antialias: false,
						preserveDrawingBuffer: true,
						alpha: false,
						premultipliedAlpha: false,
					});

					// Set renderer size - use same aspect as viewer
					const renderWidth = 1920; // Fixed render resolution
					const renderHeight = 1080;

					console.log('Renderer size:', renderWidth, 'x', renderHeight);
					renderer.setSize(renderWidth, renderHeight);

					// Add canvas to visible container
					renderer.domElement.style.maxWidth = '80vw';
					renderer.domElement.style.maxHeight = '60vh';
					renderer.domElement.style.display = 'block';
					container.appendChild(renderer.domElement);

					// Create video texture - EXACTLY like viewer
					const videoTexture = new THREE.VideoTexture(videoElement);
					videoTexture.minFilter = THREE.LinearFilter;
					videoTexture.magFilter = THREE.LinearFilter;
					videoTexture.format = THREE.RGBFormat;
					videoTexture.needsUpdate = true;

					// Create camera - EXACTLY like viewer uses perspective camera
					const planeWidth = 1;
					const aspect = 16 / 9;
					const planeHeight = planeWidth / aspect;

					// Camera setup matching the viewer's perspective
					const camera = new THREE.PerspectiveCamera(
						75, // Standard FOV like most viewers
						aspect,
						0.1,
						1000
					);
					camera.position.set(0, 0, 1.5); // Position back from plane
					camera.lookAt(0, 0, 0);

					// Helper function to render a single warped frame - EXACTLY like VideoPlane component
					const renderWarpedFrame = async (isLeft) => {
						return new Promise((resolveBlob) => {
							// Update title
							title.textContent = `Rendering ${isLeft ? 'left' : 'right'} camera...`;

							// Clear scene
							while (scene.children.length > 0) {
								const child = scene.children[0];
								scene.remove(child);
								if (child.geometry) child.geometry.dispose();
								if (child.material) child.material.dispose();
							}

							const uniforms = isLeft ? leftUniforms : rightUniforms;
							console.log(`Creating shader for ${isLeft ? 'left' : 'right'} with uniforms:`, uniforms);

							// Create plane geometry - EXACTLY like viewer
							const planeGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight);

							// Create shader material - EXACTLY like viewer
							const shader = fisheyeShader(isLeft);
							const formattedUniforms = formatUniforms(uniforms, videoTexture);

							const material = new THREE.ShaderMaterial({
								uniforms: formattedUniforms,
								vertexShader: shader.vertexShader,
								fragmentShader: shader.fragmentShader,
							});

							const plane = new THREE.Mesh(planeGeometry, material);

							// Position at origin, no rotation (for clean orthogonal capture)
							plane.position.set(0, 0, 0);
							scene.add(plane);

							// Point camera at plane
							camera.lookAt(plane.position);

							// Render loop to ensure texture is loaded
							let renderCount = 0;
							const renderLoop = () => {
								videoTexture.needsUpdate = true;
								renderer.render(scene, camera);
								renderCount++;

								if (renderCount < 10) {
									requestAnimationFrame(renderLoop);
								} else {
									// Wait for last render to complete, then capture
									setTimeout(() => {
										videoTexture.needsUpdate = true;
										renderer.render(scene, camera);

										renderer.domElement.toBlob(
											(blob) => {
												console.log(
													`Rendered ${isLeft ? 'left' : 'right'} frame:`,
													blob.size,
													'bytes'
												);
												resolveBlob(blob);
											},
											'image/png',
											1.0
										);
									}, 200);
								}
							};

							renderLoop();
						});
					};

					// Render both frames sequentially
					const leftBlob = await renderWarpedFrame(true);
					const rightBlob = await renderWarpedFrame(false);

					// Show completion message briefly
					title.textContent = 'Frames extracted successfully!';
					await new Promise((r) => setTimeout(r, 1000));

					// Cleanup
					document.body.removeChild(container);
					renderer.dispose();
					videoTexture.dispose();
					scene.clear();

					resolve({ leftBlob, rightBlob });
				} catch (error) {
					console.error('Frame extraction error:', error);
					reject(error);
				}
			},
			{ once: true }
		);

		// Handle seek errors
		videoElement.addEventListener(
			'error',
			function onError(e) {
				videoElement.removeEventListener('error', onError);
				reject(new Error('Failed to seek video: ' + e.message));
			},
			{ once: true }
		);
	});
}

/**
 * Process match calibration by extracting warped frames and sending to backend.
 *
 * @param {string} matchId - Match identifier
 * @param {HTMLVideoElement} videoElement - Video element with stacked video loaded
 * @param {Object} leftUniforms - Left camera lens uniforms
 * @param {Object} rightUniforms - Right camera lens uniforms
 * @param {number} frameTime - Time in seconds to extract frame (default: 3.33s = frame 100 at 30fps)
 * @returns {Promise<Object>} Processing result with params
 */
export async function processMatchWithWarpedFrames(
	matchId,
	videoElement,
	leftUniforms,
	rightUniforms,
	frameTime = 100 / 30 // Frame 100 at 30fps
) {
	// Extract and warp frames
	const { leftBlob, rightBlob } = await extractWarpedFrames(videoElement, frameTime, leftUniforms, rightUniforms);

	// Create form data
	const formData = new FormData();
	formData.append('left_frame', leftBlob, 'left_frame.png');
	formData.append('right_frame', rightBlob, 'right_frame.png');

	// Send to backend
	const response = await fetch(`/api/matches/${matchId}/process-with-frames`, {
		method: 'POST',
		body: formData,
	});

	if (!response.ok) {
		const error = await response.json();
		throw new Error(error.detail || 'Failed to process frames');
	}

	return await response.json();
}
