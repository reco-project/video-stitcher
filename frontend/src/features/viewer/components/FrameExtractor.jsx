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
				uniforms={formatUniforms(uniforms, texture, {}, 0)}  // blendWidth=0 for full frame capture
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
 * 
 * @param {string} videoSrc - URL of the video to extract frames from
 * @param {number} frameTime - Exact time in seconds to seek to (optional if frameTimePercent is provided)
 * @param {number} frameTimePercent - Percentage of video duration to seek to (0-1, default 0.1 = 10%)
 * @param {object} leftUniforms - Shader uniforms for left camera
 * @param {object} rightUniforms - Shader uniforms for right camera
 * @param {function} onComplete - Callback with { leftBlob, rightBlob }
 * @param {function} onError - Callback with error
 */
const FrameExtractor = ({ videoSrc, frameTime, frameTimePercent = 0.1, leftUniforms, rightUniforms, onComplete, onError }) => {
	const [phase, setPhase] = useState('loading'); // 'loading' | 'left' | 'right' | 'done'
	const [leftBlob, setLeftBlob] = useState(null);
	const [texture, setTexture] = useState(null);
	const [computedFrameTime, setComputedFrameTime] = useState(frameTime);
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
			// Use explicit frameTime if provided, otherwise calculate from percentage
			const targetTime = frameTime != null ? frameTime : Math.floor(video.duration * frameTimePercent);
			setComputedFrameTime(targetTime);
			video.currentTime = targetTime;
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
			const targetTime = frameTime != null ? frameTime : Math.floor(video.duration * frameTimePercent);
			setComputedFrameTime(targetTime);
			video.currentTime = targetTime;
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
	}, [frameTime, frameTimePercent, onError]);

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

	// Progress calculation
	const progressSteps = ['loading', 'left', 'right', 'done'];
	const currentStepIndex = progressSteps.indexOf(phase);
	const progressPercent = ((currentStepIndex + 1) / progressSteps.length) * 100;

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
			<div className="bg-card border border-border rounded-xl shadow-2xl max-w-3xl w-full mx-4 overflow-hidden">
				{/* Header */}
				<div className="px-6 py-4 border-b border-border bg-muted/30">
					<h3 className="text-lg font-semibold">Extracting Calibration Frames</h3>
					<p className="text-sm text-muted-foreground mt-1">
						Processing video at {computedFrameTime != null ? `${Math.floor(computedFrameTime / 60)}:${String(Math.floor(computedFrameTime % 60)).padStart(2, '0')}` : '...'}
					</p>
				</div>

				{/* Progress bar */}
				<div className="h-1 bg-muted">
					<div 
						className="h-full bg-primary transition-all duration-500 ease-out"
						style={{ width: `${progressPercent}%` }}
					/>
				</div>

				{/* Video preview */}
				<div className="p-4">
					<div className="w-full aspect-video bg-black rounded-lg overflow-hidden ring-1 ring-border">
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
							<div className="w-full h-full flex items-center justify-center">
								<div className="text-center">
									<div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-3" />
									<p className="text-sm text-muted-foreground">
										{phase === 'loading' ? 'Loading video...' : phase === 'done' ? 'Complete!' : 'Preparing...'}
									</p>
								</div>
							</div>
						)}
					</div>
				</div>

				{/* Step indicators */}
				<div className="px-6 pb-5">
					<div className="flex items-center justify-between">
						{[
							{ key: 'loading', label: 'Load Video' },
							{ key: 'left', label: 'Left Camera' },
							{ key: 'right', label: 'Right Camera' },
							{ key: 'done', label: 'Complete' },
						].map((step, index) => {
							const stepIndex = progressSteps.indexOf(step.key);
							const isActive = phase === step.key;
							const isComplete = currentStepIndex > stepIndex;
							
							return (
								<div key={step.key} className="flex items-center">
									<div className="flex flex-col items-center">
										<div className={`
											w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium transition-all
											${isComplete ? 'bg-primary text-primary-foreground' : ''}
											${isActive ? 'bg-primary/20 text-primary ring-2 ring-primary ring-offset-2 ring-offset-background' : ''}
											${!isActive && !isComplete ? 'bg-muted text-muted-foreground' : ''}
										`}>
											{isComplete ? '✓' : index + 1}
										</div>
										<span className={`text-xs mt-1.5 ${isActive ? 'text-primary font-medium' : 'text-muted-foreground'}`}>
											{step.label}
										</span>
									</div>
									{index < 3 && (
										<div className={`w-12 h-0.5 mx-2 mb-5 ${isComplete ? 'bg-primary' : 'bg-muted'}`} />
									)}
								</div>
							);
						})}
					</div>
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
