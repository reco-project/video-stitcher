import * as THREE from 'three';
import React, { useEffect } from 'react';
import { useViewerStore } from '../stores/store.js';
import { Canvas } from '@react-three/fiber';
import fisheyeShader from '../shaders/fisheye.js';
import { ErrorBoundary } from 'react-error-boundary';
import Controls from './Controls.jsx';
import { formatUniforms } from '../utils/utils.js';
import VideoPlayerContainer from './VideoPlayer.jsx';
import { useCustomVideoTexture } from '../hooks/useCustomVideoTexture.js';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

const ViewerErrorFallback = ({ error, resetErrorBoundary }) => {
	return (
		<div className="flex items-center justify-center p-8">
			<Alert variant="destructive" className="max-w-2xl">
				<AlertTitle>Failed to load video viewer</AlertTitle>
				<AlertDescription className="mt-2">
					<p className="mb-2">{error.message || 'An unexpected error occurred'}</p>
					<p className="text-sm text-muted-foreground mb-4">
						This may happen if the match data is incomplete or the video file cannot be loaded.
					</p>
					<Button onClick={resetErrorBoundary} variant="outline" size="sm">
						Try Again
					</Button>
				</AlertDescription>
			</Alert>
		</div>
	);
};

const VideoPlane = ({ texture, isLeft }) => {
	const selectedMatch = useViewerStore((s) => s.selectedMatch);

	if (!selectedMatch) return null;

	const params = selectedMatch.params || {};
	const u = isLeft ? selectedMatch.left_uniforms : selectedMatch.right_uniforms;

	// Validate uniforms exist
	if (!u || !u.width || !u.fx) {
		console.error('Missing uniforms for', isLeft ? 'left' : 'right', 'camera:', u);
		console.error('Match data:', selectedMatch);
		return null;
	}

	const planeWidth = 1;
	const aspect = 16 / 9;

	const position = isLeft
		? [0, 0, (planeWidth / 2) * (1 - (params.intersect || 0.5))]
		: [(planeWidth / 2) * (1 - (params.intersect || 0.5)), params.xTy || 0, 0];
	const rotation = isLeft ? [params.zRx || 0, THREE.MathUtils.degToRad(90), 0] : [0, 0, params.xRz || 0];

	return (
		<mesh position={position} rotation={rotation}>
			<planeGeometry args={[planeWidth, planeWidth / aspect]} />
			<shaderMaterial uniforms={formatUniforms(u, texture)} {...fisheyeShader(isLeft)} />
		</mesh>
	);
};

const VideoPanorama = () => {
	const selectedMatch = useViewerStore((s) => s.selectedMatch);
	const src = selectedMatch ? selectedMatch.src : null;
	if (!src) return null;

	const texture = useCustomVideoTexture(src);
	if (!texture) return null;

	return (
		<group>
			<VideoPlane texture={texture} isLeft={true} />
			<VideoPlane texture={texture} isLeft={false} />
		</group>
	);
};

const Viewer = ({ selectedMatch }) => {
	const setSelectedMatch = useViewerStore((s) => s.setSelectedMatch);

	useEffect(() => {
		setSelectedMatch(selectedMatch);
	}, [selectedMatch, setSelectedMatch]);

	// Show friendly message for unprocessed matches
	if (!selectedMatch?.params || !selectedMatch?.left_uniforms || !selectedMatch?.right_uniforms) {
		const missingItems = [];
		if (!selectedMatch?.params) missingItems.push('calibration parameters');
		if (!selectedMatch?.left_uniforms) missingItems.push('left camera uniforms');
		if (!selectedMatch?.right_uniforms) missingItems.push('right camera uniforms');

		return (
			<div className="flex items-center justify-center p-8">
				<Alert className="max-w-2xl">
					<AlertTitle>Match Needs Processing</AlertTitle>
					<AlertDescription className="mt-2">
						<p className="mb-2">
							This match is missing: {missingItems.join(', ')}.
							{!selectedMatch?.params &&
								' Processing will transcode the videos and calibrate the camera parameters needed for stitching.'}
						</p>
						<p className="text-sm text-muted-foreground">
							Go back to the match list and click &quot;
							{selectedMatch?.status === 'ready' ? 'Retry' : 'Process Now'}&quot; to{' '}
							{selectedMatch?.status === 'ready' ? 're-' : ''}process this match.
						</p>
					</AlertDescription>
				</Alert>
			</div>
		);
	}

	const defaultFOV = 75;
	const cameraAxisOffset = selectedMatch.params.cameraAxisOffset;

	return (
		<ErrorBoundary FallbackComponent={ViewerErrorFallback}>
			<VideoPlayerContainer>
				<Canvas
					camera={{
						position: [cameraAxisOffset, 0, cameraAxisOffset],
						fov: defaultFOV,
						near: 0.01,
						far: 5,
					}}
				>
					<Controls />
					<VideoPanorama />
				</Canvas>
			</VideoPlayerContainer>
		</ErrorBoundary>
	);
};

export default Viewer;
