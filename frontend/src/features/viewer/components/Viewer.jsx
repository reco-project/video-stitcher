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

const VideoPlane = ({ texture, isLeft }) => {
	const selectedMatch = useViewerStore((s) => s.selectedMatch);
	const params = selectedMatch ? selectedMatch.params : {};
	const u = selectedMatch ? selectedMatch.uniforms : {};
	const planeWidth = 1;
	const aspect = 16 / 9;

	const position = isLeft
		? [0, 0, (planeWidth / 2) * (1 - params.intersect)]
		: [(planeWidth / 2) * (1 - params.intersect), params.xTy, 0];
	const rotation = isLeft ? [params.zRx, THREE.MathUtils.degToRad(90), 0] : [0, 0, params.xRz];

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
	}, [selectedMatch, setSelectedMatch]); // TODO: missing validation

	const defaultFOV = 75;
	const cameraAxisOffset = selectedMatch.params.cameraAxisOffset; // TODO: validate. No use of "?." here to avoid silent errors.

	return (
		<ErrorBoundary fallback={<div>Error loading video panorama</div>}>
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
