import React, { useRef, useState, useCallback } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import fisheyeShader from '../shaders/fisheye.js';
import { formatUniforms } from '../utils/utils.js';
import { useCustomVideoTexture } from '../hooks/useCustomVideoTexture.js';

const ExtractorPlane = ({ texture, isLeft, uniforms, onCapture }) => {
	const { gl, scene, camera } = useThree();
	const meshRef = useRef();
	const [captureReady, setCaptureReady] = useState(false);

	// Use exact same geometry as export.js
	const planeWidth = 1;
	const planeHeight = (planeWidth * 9) / 16;

	// Position at origin for export-style capture
	const position = [0, 0, 0];
	const rotation = [0, 0, 0];

	// Trigger capture after a few render frames
	React.useEffect(() => {
		if (!texture || captureReady) return;

		let frameCount = 0;
		const maxFrames = 10;

		const renderLoop = () => {
			if (frameCount < maxFrames) {
				frameCount++;
				requestAnimationFrame(renderLoop);
			} else {
				setCaptureReady(true);
				// Small delay to ensure final render is complete
				setTimeout(() => {
					// Render one final time
					gl.render(scene, camera);

					// Capture the canvas
					gl.domElement.toBlob(
						(blob) => {
							onCapture(blob);
						},
						'image/png',
						1.0
					);
				}, 100);
			}
		};

		renderLoop();
	}, [texture, gl, scene, camera, onCapture, captureReady]);

	if (!texture || !uniforms) return null;

	return (
		<mesh ref={meshRef} position={position} rotation={rotation}>
			<planeGeometry args={[planeWidth, planeHeight]} />
			<shaderMaterial uniforms={formatUniforms(uniforms, texture)} {...fisheyeShader(isLeft)} />
		</mesh>
	);
};

const ExtractionCanvas = ({ videoSrc, isLeft, uniforms, onComplete }) => {
	const texture = useCustomVideoTexture(videoSrc);
	const [captureStarted, setCaptureStarted] = useState(false);

	const handleCapture = useCallback(
		(blob) => {
			if (!captureStarted) {
				setCaptureStarted(true);
				onComplete(blob);
			}
		},
		[captureStarted, onComplete]
	);

	// Fixed render resolution matching the video half-frame
	const renderWidth = 1920;
	const renderHeight = 1080;

	// Match export.js geometry exactly
	const planeWidth = 1;
	const planeHeight = (planeWidth * 9) / 16;

	// Match export.js camera FOV formula exactly
	const cameraFOV = Math.atan(planeHeight / 2 / 1) * (180 / Math.PI) * 2;
	const cameraDistance = 1;

	return (
		<Canvas
			gl={{
				preserveDrawingBuffer: true,
				antialias: false,
				alpha: false,
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
			<ExtractorPlane texture={texture} isLeft={isLeft} uniforms={uniforms} onCapture={handleCapture} />
		</Canvas>
	);
};

const FrameExtractor = ({ videoSrc, frameTime, leftUniforms, rightUniforms, onComplete, onError }) => {
	const [currentSide, setCurrentSide] = useState('left');
	const [leftBlob, setLeftBlob] = useState(null);
	const videoRef = useRef(null);
	const [videoReady, setVideoReady] = useState(false);

	// Handle video seeking
	React.useEffect(() => {
		if (!videoRef.current) return;

		const video = videoRef.current;

		const handleSeeked = () => {
			console.log('Video seeked to:', video.currentTime);
			setVideoReady(true);
		};

		const handleError = (e) => {
			console.error('Video error:', e);
			onError?.(new Error('Failed to load or seek video'));
		};

		video.addEventListener('seeked', handleSeeked);
		video.addEventListener('error', handleError);

		// Seek to frame time
		video.currentTime = frameTime;

		return () => {
			video.removeEventListener('seeked', handleSeeked);
			video.removeEventListener('error', handleError);
		};
	}, [frameTime, onError]);

	const handleLeftComplete = useCallback((blob) => {
		console.log('Left frame captured:', blob.size, 'bytes');
		setLeftBlob(blob);
		setCurrentSide('right');
	}, []);

	const handleRightComplete = useCallback(
		(blob) => {
			console.log('Right frame captured:', blob.size, 'bytes');
			if (leftBlob) {
				onComplete?.({ leftBlob, rightBlob: blob });
			}
		},
		[leftBlob, onComplete]
	);

	return (
		<div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90">
			<div className="bg-gray-900 p-6 rounded-lg border-2 border-emerald-600 max-w-4xl w-full">
				<div className="text-white font-bold mb-4">
					Extracting and warping frames...
					<span className="ml-2 text-emerald-400">
						{currentSide === 'left' ? 'Left camera' : 'Right camera'}
					</span>
				</div>

				<div className="w-full aspect-video bg-black rounded overflow-hidden">
					{videoReady && currentSide === 'left' && (
						<ExtractionCanvas
							videoSrc={videoSrc}
							isLeft={true}
							uniforms={leftUniforms}
							onComplete={handleLeftComplete}
						/>
					)}
					{videoReady && currentSide === 'right' && (
						<ExtractionCanvas
							videoSrc={videoSrc}
							isLeft={false}
							uniforms={rightUniforms}
							onComplete={handleRightComplete}
						/>
					)}
				</div>

				{/* Hidden video element for seeking */}
				<video
					ref={videoRef}
					src={videoSrc}
					crossOrigin="anonymous"
					preload="auto"
					style={{ display: 'none' }}
				/>
			</div>
		</div>
	);
};

export default FrameExtractor;
