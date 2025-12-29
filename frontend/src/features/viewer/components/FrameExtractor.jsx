import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import fisheyeShader from '../shaders/fisheye.js';
import { formatUniforms } from '../utils/utils.js';

/**
 * Single plane that renders either left or right view based on props.
 * The shader handles selecting the correct half of the stacked video.
 */
const ExtractorPlane = ({ texture, isLeft, uniforms }) => {
	const meshRef = useRef();

	// Use exact same geometry as export.js
	const planeWidth = 1;
	const planeHeight = (planeWidth * 9) / 16;

	// Force texture update on every frame
	useFrame(() => {
		if (texture) {
			texture.needsUpdate = true;
		}
	});

	if (!texture || !uniforms) return null;

	return (
		<mesh ref={meshRef} position={[0, 0, 0]} rotation={[0, 0, 0]}>
			<planeGeometry args={[planeWidth, planeHeight]} />
			<shaderMaterial
				key={isLeft ? 'left' : 'right'} // Force shader recreation when side changes
				uniforms={formatUniforms(uniforms, texture)}
				{...fisheyeShader(isLeft)}
			/>
		</mesh>
	);
};

/**
 * Capture controller - handles the capture logic inside the Canvas context
 */
const CaptureController = ({ texture, onCapture, shouldCapture, side }) => {
	const { gl, scene, camera } = useThree();
	const capturedRef = useRef(false);
	const frameCountRef = useRef(0);

	useEffect(() => {
		// Reset when side changes
		capturedRef.current = false;
		frameCountRef.current = 0;
	}, [side]);

	useFrame(() => {
		if (!shouldCapture || capturedRef.current || !texture) return;

		// Update texture
		texture.needsUpdate = true;

		// Wait for enough frames to render
		frameCountRef.current++;
		if (frameCountRef.current < 30) return;

		// Capture
		capturedRef.current = true;

		// Small delay then capture
		setTimeout(() => {
			texture.needsUpdate = true;
			gl.render(scene, camera);

			gl.domElement.toBlob(
				(blob) => {
					if (blob) {
						onCapture(blob, side);
					} else {
						console.error(`Failed to create ${side} blob`);
						onCapture(null, side);
					}
				},
				'image/png',
				1.0
			);
		}, 100);
	});

	return null;
};

/**
 * Main FrameExtractor component.
 * Uses a SINGLE persistent Canvas to avoid WebGL context loss.
 */
const FrameExtractor = ({ videoSrc, frameTime, leftUniforms, rightUniforms, onComplete, onError }) => {
	const [phase, setPhase] = useState('loading'); // 'loading' | 'left' | 'right' | 'done'
	const [leftBlob, setLeftBlob] = useState(null);
	const [texture, setTexture] = useState(null);
	const videoRef = useRef(null);

	// Create texture from video when ready
	useEffect(() => {
		const video = videoRef.current;
		if (!video) return;

		let videoTexture = null;

		const createTexture = () => {
			if (video.readyState >= 3 && video.videoWidth > 0 && !videoTexture) {
				videoTexture = new THREE.VideoTexture(video);
				videoTexture.minFilter = THREE.LinearFilter;
				videoTexture.magFilter = THREE.LinearFilter;
				videoTexture.generateMipmaps = false;
				videoTexture.needsUpdate = true;
				setTexture(videoTexture);
				setPhase('left');
			}
		};

		const handleSeeked = () => {
			video.pause();
			createTexture();
		};

		const handleCanplay = () => {
			createTexture();
		};

		const handleLoadedMetadata = () => {
			video.pause();
			video.currentTime = frameTime;
		};

		const handleError = (e) => {
			console.error('Video error:', e);
			onError?.(new Error('Failed to load video'));
		};

		video.addEventListener('loadedmetadata', handleLoadedMetadata);
		video.addEventListener('canplay', handleCanplay);
		video.addEventListener('seeked', handleSeeked);
		video.addEventListener('error', handleError);

		// If already loaded, seek
		if (video.readyState >= 1) {
			video.pause();
			video.currentTime = frameTime;
		}

		return () => {
			video.removeEventListener('loadedmetadata', handleLoadedMetadata);
			video.removeEventListener('canplay', handleCanplay);
			video.removeEventListener('seeked', handleSeeked);
			video.removeEventListener('error', handleError);
			if (videoTexture) {
				videoTexture.dispose();
			}
		};
	}, [frameTime, onError]);

	// Handle frame capture
	const handleCapture = useCallback(
		(blob, side) => {
			if (!blob) {
				onError?.(new Error(`Failed to capture ${side} frame`));
				return;
			}

			if (side === 'left') {
				setLeftBlob(blob);
				// Wait a bit then switch to right
				setTimeout(() => {
					setPhase('right');
				}, 300);
			} else if (side === 'right') {
				setPhase('done');
				// Complete with both blobs
				onComplete?.({ leftBlob, rightBlob: blob });
			}
		},
		[leftBlob, onComplete, onError]
	);

	// Render dimensions
	const renderWidth = 1920;
	const renderHeight = 1080;

	// Camera setup matching export.js
	const planeWidth = 1;
	const planeHeight = (planeWidth * 9) / 16;
	const cameraFOV = Math.atan(planeHeight / 2 / 1) * (180 / Math.PI) * 2;
	const cameraDistance = 1;

	const isLeft = phase === 'left';
	const uniforms = isLeft ? leftUniforms : rightUniforms;
	const shouldCapture = phase === 'left' || phase === 'right';

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
			<div className="bg-gray-900 p-6 rounded-lg border-2 border-emerald-600 max-w-4xl w-full">
				<div className="text-white font-bold mb-4">
					Extracting and warping frames...
					<span className="ml-2 text-emerald-400">
						{phase === 'loading'
							? 'Loading video...'
							: phase === 'left'
								? 'Left camera'
								: phase === 'right'
									? 'Right camera'
									: 'Complete'}
					</span>
				</div>

				<div className="w-full aspect-video bg-black rounded overflow-hidden">
					{texture && shouldCapture ? (
						<Canvas
							gl={{
								preserveDrawingBuffer: true,
								antialias: false,
								alpha: false,
								powerPreference: 'high-performance',
							}}
							camera={{
								position: [0, 0, cameraDistance],
								fov: cameraFOV,
								aspect: 16 / 9,
								near: 0.01,
								far: 5,
							}}
							style={{ width: '100%', height: '100%' }}
							dpr={1}
							onCreated={({ gl }) => {
								gl.setSize(renderWidth, renderHeight, false);
							}}
						>
							<ExtractorPlane texture={texture} isLeft={isLeft} uniforms={uniforms} />
							<CaptureController
								texture={texture}
								onCapture={handleCapture}
								shouldCapture={shouldCapture}
								side={isLeft ? 'left' : 'right'}
							/>
						</Canvas>
					) : (
						<div
							style={{
								width: '100%',
								height: '100%',
								background: '#000',
								display: 'flex',
								alignItems: 'center',
								justifyContent: 'center',
								color: '#888',
							}}
						>
							{phase === 'loading' ? 'Loading video...' : phase === 'done' ? 'Done!' : 'Preparing...'}
						</div>
					)}
				</div>

				{/* Hidden video element */}
				<video
					ref={videoRef}
					src={videoSrc}
					crossOrigin="anonymous"
					preload="auto"
					muted
					playsInline
					style={{ display: 'none' }}
				/>
			</div>
		</div>
	);
};

export default FrameExtractor;
