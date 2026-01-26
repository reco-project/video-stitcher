import * as THREE from 'three';
import React, { useEffect, useState, useRef, useCallback } from 'react';
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
import { updateMatch } from '@/features/matches/api/matches.js';
import { CameraProvider, useCameraControls } from '../stores/cameraContext';
import RecalibratePanel from './RecalibratePanel';
import { DualColorCorrectionPanel } from './ColorCorrectionPanel.jsx';
import DescriptionPanel from './DescriptionPanel.jsx';
import VideoTitle from './VideoTitle.jsx';

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

const CameraControlsWrapper = ({ yawRange, pitchRange, children }) => {
	const { setYawRange, setPitchRange } = useCameraControls();

	useEffect(() => {
		setYawRange(yawRange);
		setPitchRange(pitchRange);
	}, [yawRange, pitchRange, setYawRange, setPitchRange]);

	return <>{children}</>;
};

const VideoPlane = ({ texture, isLeft }) => {
	const selectedMatch = useViewerStore((s) => s.selectedMatch);
	const leftColorCorrection = useViewerStore((s) => s.leftColorCorrection);
	const rightColorCorrection = useViewerStore((s) => s.rightColorCorrection);

	if (!selectedMatch) return null;

	const params = selectedMatch.params || {};
	const u = isLeft ? selectedMatch.left_uniforms : selectedMatch.right_uniforms;
	const colorCorrection = isLeft ? leftColorCorrection : rightColorCorrection;

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

	// Generate key from color correction to force shader material update
	const ccKey = JSON.stringify(colorCorrection);

	return (
		<mesh position={position} rotation={rotation}>
			<planeGeometry args={[planeWidth, planeWidth / aspect]} />
			<shaderMaterial 
				key={ccKey}
				uniforms={formatUniforms(u, texture, colorCorrection)} 
				{...fisheyeShader(isLeft)} 
			/>
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
	const videoRef = useViewerStore((s) => s.videoRef);
	const leftColorCorrection = useViewerStore((s) => s.leftColorCorrection);
	const rightColorCorrection = useViewerStore((s) => s.rightColorCorrection);
	const setLeftColorCorrection = useViewerStore((s) => s.setLeftColorCorrection);
	const setRightColorCorrection = useViewerStore((s) => s.setRightColorCorrection);
	const resetColorCorrection = useViewerStore((s) => s.resetColorCorrection);
	const loadColorCorrectionFromMatch = useViewerStore((s) => s.loadColorCorrectionFromMatch);

	const [yawRange, setYawRange] = useState(140);
	const [pitchRange, setPitchRange] = useState(20);
	const [saveStatus, setSaveStatus] = useState(null); // 'saving', 'success', 'error'
	const saveTimeoutRef = React.useRef(null);
	const colorSaveTimeoutRef = React.useRef(null);

	// Handler for when recalibration completes
	const handleRecalibrated = useCallback((result) => {
		// Update the local match data with new calibration params
		if (result && result.params) {
			setSelectedMatch({
				...selectedMatch,
				params: result.params,
				num_matches: result.num_matches,
			});
		}
	}, [selectedMatch, setSelectedMatch]);

	useEffect(() => {
		setSelectedMatch(selectedMatch);
	}, [selectedMatch, setSelectedMatch]);

	// Load color correction from match when it changes
	useEffect(() => {
		loadColorCorrectionFromMatch();
	}, [selectedMatch?.id, loadColorCorrectionFromMatch]);

	// Mark match as viewed when component mounts
	useEffect(() => {
		if (selectedMatch && selectedMatch.id && !selectedMatch.viewed) {
			try {
				// Update backend - only send the viewed field
				updateMatch(selectedMatch.id, { id: selectedMatch.id, viewed: true });

				// Update localStorage for MatchCard badge
				const viewedMatches = JSON.parse(localStorage.getItem('viewedMatches') || '[]');
				if (!viewedMatches.includes(selectedMatch.id)) {
					viewedMatches.push(selectedMatch.id);
					localStorage.setItem('viewedMatches', JSON.stringify(viewedMatches));
				}
			} catch (err) {
				console.warn('Failed to mark match as viewed:', err);
			}
		}
	}, [selectedMatch?.id]);

	// Load panning ranges from match metadata if available
	// Track initial load to prevent reset after save
	const initialLoadRef = useRef(true);
	const lastLoadedMatchIdRef = useRef(null);

	useEffect(() => {
		// Load ranges when match changes or when component mounts with a match
		if (selectedMatch?.id) {
			// Check if we need to load (new match ID or component just mounted)
			const needsLoad = selectedMatch.id !== lastLoadedMatchIdRef.current;

			if (needsLoad) {
				if (selectedMatch?.metadata?.panningRanges) {
					setYawRange(selectedMatch.metadata.panningRanges.yaw || 140);
					setPitchRange(selectedMatch.metadata.panningRanges.pitch || 20);
				} else {
					// Reset to defaults if no saved ranges
					setYawRange(140);
					setPitchRange(20);
				}
				lastLoadedMatchIdRef.current = selectedMatch.id;
			}
			// After first load, mark as not initial
			initialLoadRef.current = false;
		}

		// Cleanup: reset refs when component unmounts so ranges reload on remount
		return () => {
			lastLoadedMatchIdRef.current = null;
			initialLoadRef.current = true;
		};
	}, [selectedMatch?.id, selectedMatch?.metadata?.panningRanges]);

	// Auto-save panning ranges with debouncing
	useEffect(() => {
		// Don't save on initial load, if no match selected, or if match doesn't exist on backend yet
		if (!selectedMatch?.id || initialLoadRef.current || !selectedMatch?.params) return;

		// Skip save if values are the same as stored (prevents save on load)
		if (
			selectedMatch.metadata?.panningRanges?.yaw === yawRange &&
			selectedMatch.metadata?.panningRanges?.pitch === pitchRange
		) {
			return;
		}

		// Clear existing timeout
		if (saveTimeoutRef.current) {
			clearTimeout(saveTimeoutRef.current);
		}

		// Set new timeout for debounced save
		saveTimeoutRef.current = setTimeout(async () => {
			try {
				setSaveStatus('saving');
				// Only send the metadata field being updated
				const updatedMatch = {
					id: selectedMatch.id,
					metadata: {
						...selectedMatch.metadata,
						panningRanges: {
							yaw: yawRange,
							pitch: pitchRange,
						},
					},
				};
				await updateMatch(selectedMatch.id, updatedMatch);
				setSaveStatus('success');
				// Clear success message after 2 seconds
				setTimeout(() => setSaveStatus(null), 2000);
			} catch (err) {
				console.warn('Failed to auto-save panning ranges:', err);
				setSaveStatus('error');
				setTimeout(() => setSaveStatus(null), 3000);
			}
		}, 1000); // Debounce for 1 second

		return () => {
			if (saveTimeoutRef.current) {
				clearTimeout(saveTimeoutRef.current);
			}
		};
	}, [yawRange, pitchRange, selectedMatch]);

	// Auto-save color correction with debouncing
	useEffect(() => {
		// Don't save on initial load, if no match selected, or if match doesn't exist on backend yet
		if (!selectedMatch?.id || initialLoadRef.current || !selectedMatch?.params) return;

		// Clear existing timeout
		if (colorSaveTimeoutRef.current) {
			clearTimeout(colorSaveTimeoutRef.current);
		}

		// Set new timeout for debounced save
		colorSaveTimeoutRef.current = setTimeout(async () => {
			try {
				setSaveStatus('saving');
				const updatedMatch = {
					id: selectedMatch.id,
					metadata: {
						...selectedMatch.metadata,
						colorCorrection: {
							left: leftColorCorrection,
							right: rightColorCorrection,
						},
					},
				};
				await updateMatch(selectedMatch.id, updatedMatch);
				setSaveStatus('success');
				setTimeout(() => setSaveStatus(null), 2000);
			} catch (err) {
				console.warn('Failed to auto-save color correction:', err);
				setSaveStatus('error');
				setTimeout(() => setSaveStatus(null), 3000);
			}
		}, 1000);

		return () => {
			if (colorSaveTimeoutRef.current) {
				clearTimeout(colorSaveTimeoutRef.current);
			}
		};
	}, [leftColorCorrection, rightColorCorrection, selectedMatch]);

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
		<div className="w-full flex flex-col items-center gap-3 px-4 py-4">
			{/* Video Title */}
			<VideoTitle match={selectedMatch} />

			{/* 3D Viewer - Takes full width of parent */}
			<ErrorBoundary FallbackComponent={ViewerErrorFallback}>
				<CameraProvider>
					<CameraControlsWrapper yawRange={yawRange} pitchRange={pitchRange}>
						<VideoPlayerContainer>
							<Canvas
								frameloop="always"
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
					</CameraControlsWrapper>
				</CameraProvider>
			</ErrorBoundary>

			{/* Description Panel - Collapsible */}
			<div className="w-full max-w-6xl">
				<DescriptionPanel
					match={selectedMatch}
					yawRange={yawRange}
					pitchRange={pitchRange}
					onYawChange={setYawRange}
					onPitchChange={setPitchRange}
					saveStatus={saveStatus}
				/>
			</div>
			
			{/* Color Correction - Collapsible */}
			<div className="w-full max-w-6xl">
				<DualColorCorrectionPanel
					leftValues={leftColorCorrection}
					rightValues={rightColorCorrection}
					onLeftChange={setLeftColorCorrection}
					onRightChange={setRightColorCorrection}
					onResetAll={resetColorCorrection}
					matchId={selectedMatch?.id}
					videoRef={videoRef}
				/>
			</div>

			{/* Recalibrate - Collapsible */}
			<div className="w-full max-w-6xl">
				<RecalibratePanel
					match={selectedMatch}
					videoRef={videoRef}
					onRecalibrated={handleRecalibrated}
				/>
			</div>
		</div>
	);
};

export default Viewer;
