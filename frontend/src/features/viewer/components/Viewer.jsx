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
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/toast';
import { updateMatch } from '@/features/matches/api/matches.js';
import { getProfile } from '@/features/profiles/api/profiles.js';
import ProfileCombobox from '@/features/matches/components/ProfileCombobox.jsx';
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
	const blendWidth = useViewerStore((s) => s.blendWidth);

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

	// Generate key from params, color correction, and blend width to force re-render on changes
	const meshKey = JSON.stringify({ params, colorCorrection, blendWidth });

	// Left plane: fully opaque, renders first as base layer
	// Right plane: transparent with alpha blend, renders on top
	const isTransparent = !isLeft && blendWidth > 0;

	return (
		<mesh key={meshKey} position={position} rotation={rotation} renderOrder={isLeft ? 1 : 2}>
			<planeGeometry args={[planeWidth, planeWidth / aspect]} />
			<shaderMaterial
				uniforms={formatUniforms(u, texture, colorCorrection, blendWidth)}
				transparent={isTransparent}
				depthWrite={!isTransparent}
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
	const { showToast } = useToast();

	const [activeMatch, setActiveMatch] = useState(selectedMatch);

	const [yawRange, setYawRange] = useState(140);
	const [pitchRange, setPitchRange] = useState(20);
	const [saveStatus, setSaveStatus] = useState(null); // 'saving', 'success', 'error'
	const saveTimeoutRef = React.useRef(null);
	const colorSaveTimeoutRef = React.useRef(null);
	const [liveLeftProfileId, setLiveLeftProfileId] = useState('');
	const [liveRightProfileId, setLiveRightProfileId] = useState('');
	const [liveUpdatingSide, setLiveUpdatingSide] = useState(null);
	const isLiveMatch = activeMatch?.id === 'live';
	const defaultLiveParams = {
		cameraAxisOffset: 0.7,
		intersect: 0.5,
		xTy: 0.0,
		xRz: 0.0,
		zRx: 0.0,
	};
	const defaultLiveUniforms = {
		width: 1920,
		height: 1080,
		fx: 1000,
		fy: 1000,
		cx: 960,
		cy: 540,
		d: [0, 0, 0, 0],
	};
	const liveDefaultsAppliedRef = useRef(false);

	const buildUniforms = (profile) => {
		if (
			!profile?.resolution ||
			typeof profile.resolution.width !== 'number' ||
			typeof profile.resolution.height !== 'number'
		) {
			throw new Error(`Profile ${profile?.id || 'unknown'} has invalid resolution`);
		}

		if (
			!profile.camera_matrix ||
			typeof profile.camera_matrix.fx !== 'number' ||
			typeof profile.camera_matrix.fy !== 'number' ||
			typeof profile.camera_matrix.cx !== 'number' ||
			typeof profile.camera_matrix.cy !== 'number'
		) {
			throw new Error(`Profile ${profile?.id || 'unknown'} has invalid camera matrix`);
		}

		if (
			!profile.distortion_coeffs ||
			!Array.isArray(profile.distortion_coeffs) ||
			profile.distortion_coeffs.length !== 4 ||
			!profile.distortion_coeffs.every((c) => typeof c === 'number')
		) {
			throw new Error(`Profile ${profile?.id || 'unknown'} has invalid distortion coefficients`);
		}

		return {
			width: profile.resolution.width,
			height: profile.resolution.height,
			fx: profile.camera_matrix.fx,
			fy: profile.camera_matrix.fy,
			cx: profile.camera_matrix.cx,
			cy: profile.camera_matrix.cy,
			d: profile.distortion_coeffs,
		};
	};

	// Handler for when recalibration completes
	const handleRecalibrated = useCallback(
		(result) => {
			// Only update match data if calibration succeeded
			// If calibration failed, keep existing params
			if (result && result.params && !result.calibration_failed) {
				const updated = {
					...activeMatch,
					params: result.params,
					num_matches: result.num_matches,
					confidence: result.confidence,
					status: 'ready',
				};
				setActiveMatch(updated);
				setSelectedMatch(updated);
			}
			// If calibration failed, RecalibratePanel will show the warning
		},
		[activeMatch, setSelectedMatch]
	);

	useEffect(() => {
		setActiveMatch(selectedMatch);
	}, [selectedMatch]);

	useEffect(() => {
		if (activeMatch) {
			setSelectedMatch(activeMatch);
		}
	}, [activeMatch, setSelectedMatch]);

	// Load color correction from match when it changes
	useEffect(() => {
		loadColorCorrectionFromMatch();
	}, [activeMatch?.id, loadColorCorrectionFromMatch]);

	// Mark match as viewed when component mounts
	useEffect(() => {
		if (activeMatch && activeMatch.id && !activeMatch.viewed) {
			try {
				// Update backend - only send the viewed field
				updateMatch(activeMatch.id, { id: activeMatch.id, viewed: true });

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
	}, [activeMatch?.id]);

	useEffect(() => {
		if (!isLiveMatch) return;
		setLiveLeftProfileId(activeMatch?.metadata?.left_profile_id || activeMatch?.left_videos?.[0]?.profile_id || '');
		setLiveRightProfileId(
			activeMatch?.metadata?.right_profile_id || activeMatch?.right_videos?.[0]?.profile_id || ''
		);
	}, [
		isLiveMatch,
		activeMatch?.metadata?.left_profile_id,
		activeMatch?.metadata?.right_profile_id,
		activeMatch?.left_videos,
		activeMatch?.right_videos,
	]);

	useEffect(() => {
		if (!isLiveMatch || !activeMatch?.id || liveDefaultsAppliedRef.current) return;

		const needsParams = !activeMatch?.params;
		const needsLeft = !activeMatch?.left_uniforms;
		const needsRight = !activeMatch?.right_uniforms;
		if (!needsParams && !needsLeft && !needsRight) return;

		const updatedMatch = {
			...activeMatch,
			params: activeMatch.params || defaultLiveParams,
			left_uniforms: activeMatch.left_uniforms || defaultLiveUniforms,
			right_uniforms: activeMatch.right_uniforms || defaultLiveUniforms,
		};

		liveDefaultsAppliedRef.current = true;
		setActiveMatch(updatedMatch);
		setSelectedMatch(updatedMatch);
		updateMatch(activeMatch.id, {
			id: activeMatch.id,
			params: updatedMatch.params,
			left_uniforms: updatedMatch.left_uniforms,
			right_uniforms: updatedMatch.right_uniforms,
		}).catch(() => {});
	}, [
		isLiveMatch,
		activeMatch?.id,
		activeMatch?.params,
		activeMatch?.left_uniforms,
		activeMatch?.right_uniforms,
		setSelectedMatch,
	]);

	const updateLiveProfile = useCallback(
		async (side, profileId) => {
			if (!activeMatch || !profileId) return;
			setLiveUpdatingSide(side);
			try {
				const profile = await getProfile(profileId);
				const uniforms = buildUniforms(profile);
				const updatedMetadata = {
					...activeMatch.metadata,
					...(side === 'left' ? { left_profile_id: profile.id } : { right_profile_id: profile.id }),
				};
				const updatedMatch = {
					...activeMatch,
					...(side === 'left' ? { left_uniforms: uniforms } : { right_uniforms: uniforms }),
					metadata: updatedMetadata,
				};
				setActiveMatch(updatedMatch);
				setSelectedMatch(updatedMatch);
				await updateMatch(activeMatch.id, {
					id: activeMatch.id,
					...(side === 'left' ? { left_uniforms: uniforms } : { right_uniforms: uniforms }),
					metadata: updatedMetadata,
				});
				showToast({ message: `Live ${side} profile updated`, type: 'success' });
			} catch (err) {
				showToast({ message: err.message || 'Failed to update live profile', type: 'error' });
			} finally {
				setLiveUpdatingSide(null);
			}
		},
		[activeMatch, buildUniforms, setSelectedMatch, showToast]
	);

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
	}, [activeMatch?.id, activeMatch?.metadata?.panningRanges]);

	// Auto-save panning ranges with debouncing
	useEffect(() => {
		// Don't save on initial load, if no match selected, or if match doesn't exist on backend yet
		if (!activeMatch?.id || initialLoadRef.current || !activeMatch?.params) return;

		// Skip save if values are the same as stored (prevents save on load)
		if (
			activeMatch.metadata?.panningRanges?.yaw === yawRange &&
			activeMatch.metadata?.panningRanges?.pitch === pitchRange
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
					id: activeMatch.id,
					metadata: {
						...activeMatch.metadata,
						panningRanges: {
							yaw: yawRange,
							pitch: pitchRange,
						},
					},
				};
				await updateMatch(activeMatch.id, updatedMatch);
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
	}, [yawRange, pitchRange, activeMatch]);

	// Auto-save color correction with debouncing
	useEffect(() => {
		// Don't save on initial load, if no match selected, or if match doesn't exist on backend yet
		if (!activeMatch?.id || initialLoadRef.current || !activeMatch?.params) return;

		// Clear existing timeout
		if (colorSaveTimeoutRef.current) {
			clearTimeout(colorSaveTimeoutRef.current);
		}

		// Set new timeout for debounced save
		colorSaveTimeoutRef.current = setTimeout(async () => {
			try {
				setSaveStatus('saving');
				const updatedMatch = {
					id: activeMatch.id,
					metadata: {
						...activeMatch.metadata,
						colorCorrection: {
							left: leftColorCorrection,
							right: rightColorCorrection,
						},
					},
				};
				await updateMatch(activeMatch.id, updatedMatch);
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
	}, [leftColorCorrection, rightColorCorrection, activeMatch]);

	// Show friendly message for unprocessed matches
	const isMissingData = !activeMatch?.params || !activeMatch?.left_uniforms || !activeMatch?.right_uniforms;

	if (isMissingData && !isLiveMatch) {
		const missingItems = [];
		if (!activeMatch?.params) missingItems.push('calibration parameters');
		if (!activeMatch?.left_uniforms) missingItems.push('left camera uniforms');
		if (!activeMatch?.right_uniforms) missingItems.push('right camera uniforms');

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
							{activeMatch?.status === 'ready' ? 'Retry' : 'Process Now'}&quot; to{' '}
							{activeMatch?.status === 'ready' ? 're-' : ''}process this match.
						</p>
					</AlertDescription>
				</Alert>
			</div>
		);
	}

	const defaultFOV = 75;
	const cameraAxisOffset = activeMatch?.params?.cameraAxisOffset ?? 0.7;

	return (
		<div className="w-full flex flex-col items-center gap-3 px-4 py-4">
			{/* Warning Banner for failed calibration */}
			{activeMatch.status === 'warning' && (
				<div className="w-full max-w-6xl">
					<Alert className="border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
						<AlertTitle className="text-yellow-700 dark:text-yellow-400">⚠️ Calibration Failed</AlertTitle>
						<AlertDescription className="text-yellow-600 dark:text-yellow-300">
							{activeMatch.processing?.message ||
								'Could not find enough features to calibrate cameras. Using default alignment.'}{' '}
							Use the Recalibrate panel below to try again with a different frame (look for frames with
							visible grass, textures, or distinct features).
						</AlertDescription>
					</Alert>
				</div>
			)}

			{/* Video Title */}
			<VideoTitle match={activeMatch} />

			{isLiveMatch && (
				<div className="w-full max-w-6xl">
					<div className="bg-card border rounded-lg shadow-sm p-4">
						<div className="flex flex-wrap items-center justify-between gap-3">
							<div>
								<h4 className="font-semibold">Live Profiles</h4>
								<p className="text-xs text-muted-foreground">
									Select lens profiles for the live stream. Recalibrate anytime after changes.
								</p>
							</div>
							{isMissingData && (
								<span className="text-xs text-amber-600">Select profiles to start rendering.</span>
							)}
						</div>
						<div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
							<div className="space-y-1">
								<Label className="text-xs">Left Profile</Label>
								<ProfileCombobox
									value={liveLeftProfileId}
									onChange={(profileId) => {
										setLiveLeftProfileId(profileId);
										updateLiveProfile('left', profileId);
									}}
									disabled={liveUpdatingSide === 'left'}
								/>
							</div>
							<div className="space-y-1">
								<Label className="text-xs">Right Profile</Label>
								<ProfileCombobox
									value={liveRightProfileId}
									onChange={(profileId) => {
										setLiveRightProfileId(profileId);
										updateLiveProfile('right', profileId);
									}}
									disabled={liveUpdatingSide === 'right'}
								/>
							</div>
						</div>
					</div>
				</div>
			)}

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
					match={activeMatch}
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
					matchId={activeMatch?.id}
					videoRef={videoRef}
				/>
			</div>

			{/* Recalibrate - Collapsible */}
			<div className="w-full max-w-6xl">
				<RecalibratePanel match={activeMatch} videoRef={videoRef} onRecalibrated={handleRecalibrated} />
			</div>
		</div>
	);
};

export default Viewer;
