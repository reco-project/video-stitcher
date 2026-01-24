import * as THREE from 'three';
import React, { useEffect, useState, useRef } from 'react';
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
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { ChevronDown } from 'lucide-react';
import { getProcessingDuration, getTranscodeMetrics, getQualitySettings } from '@/lib/matchHelpers.js';

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
	const [yawRange, setYawRange] = useState(140);
	const [pitchRange, setPitchRange] = useState(20);
	const [isExpanded, setIsExpanded] = useState(false);
	const [saveStatus, setSaveStatus] = useState(null); // 'saving', 'success', 'error'
	const saveTimeoutRef = React.useRef(null);

	useEffect(() => {
		setSelectedMatch(selectedMatch);
	}, [selectedMatch, setSelectedMatch]);

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
		// Don't save on initial load or if no match selected
		if (!selectedMatch?.id || initialLoadRef.current) return;

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
		<div className="w-full flex flex-col items-center gap-4 px-4 py-4">
			{/* 3D Viewer - Takes full width of parent */}
			<ErrorBoundary FallbackComponent={ViewerErrorFallback}>
				<CameraProvider>
					<CameraControlsWrapper yawRange={yawRange} pitchRange={pitchRange}>
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
					</CameraControlsWrapper>
				</CameraProvider>
			</ErrorBoundary>

			{/* Minimizable Info Panel */}
			<div className="w-full max-w-6xl bg-card border rounded-lg overflow-hidden">
				{/* Header - Always Visible */}
				<button
					onClick={() => setIsExpanded(!isExpanded)}
					className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors"
				>
					<div className="flex items-center gap-2">
						<h3 className="font-semibold text-sm">{selectedMatch.name || selectedMatch.label}</h3>
						<span className="text-xs text-green-600 font-medium">Ready</span>
						{/* Save Status Indicator */}
						{saveStatus === 'saving' && (
							<span className="text-xs text-blue-600 font-medium animate-pulse">Saving...</span>
						)}
						{saveStatus === 'success' && (
							<span className="text-xs text-green-600 font-medium">✓ Saved</span>
						)}
						{saveStatus === 'error' && (
							<span className="text-xs text-red-600 font-medium">✗ Save failed</span>
						)}
					</div>
					<ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
				</button>

				{/* Expandable Content */}
				{isExpanded && (
					<div className="border-t px-3 py-4 space-y-4 bg-muted/20">
						{/* Match Info Grid */}
						<div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
							<div>
								<span className="text-muted-foreground">Match ID:</span>
								<span className="ml-2 font-mono">{selectedMatch.id}</span>
							</div>
							<div>
								<span className="text-muted-foreground">Created:</span>
								<span className="ml-2">
									{selectedMatch.created_at
										? new Date(selectedMatch.created_at).toLocaleDateString()
										: 'N/A'}
								</span>
							</div>
							{selectedMatch.left_videos && selectedMatch.left_videos[0]?.profile_id && (
								<div className="col-span-2">
									<span className="text-muted-foreground">Left Profile:</span>
									<span className="ml-2 font-mono text-[10px] break-all">
										{selectedMatch.left_videos[0].profile_id}
									</span>
								</div>
							)}
							{selectedMatch.right_videos && selectedMatch.right_videos[0]?.profile_id && (
								<div className="col-span-2">
									<span className="text-muted-foreground">Right Profile:</span>
									<span className="ml-2 font-mono text-[10px] break-all">
										{selectedMatch.right_videos[0].profile_id}
									</span>
								</div>
							)}
							{!selectedMatch.left_videos?.[0]?.profile_id && selectedMatch.metadata?.left_profile_id && (
								<div className="col-span-2">
									<span className="text-muted-foreground">Left Profile:</span>
									<span className="ml-2 font-mono text-[10px] break-all">
										{selectedMatch.metadata.left_profile_id}
									</span>
								</div>
							)}
							{!selectedMatch.right_videos?.[0]?.profile_id &&
								selectedMatch.metadata?.right_profile_id && (
									<div className="col-span-2">
										<span className="text-muted-foreground">Right Profile:</span>
										<span className="ml-2 font-mono text-[10px] break-all">
											{selectedMatch.metadata.right_profile_id}
										</span>
									</div>
								)}
							{(() => {
								const metrics = getTranscodeMetrics(selectedMatch);
								return (
									metrics.offsetSeconds !== undefined &&
									metrics.offsetSeconds !== null && (
										<div>
											<span className="text-muted-foreground">Audio Offset:</span>
											<span className="ml-2">{metrics.offsetSeconds.toFixed(3)}s</span>
										</div>
									)
								);
							})()}
							{selectedMatch.num_matches && (
								<div>
									<span className="text-muted-foreground">Feature Matches:</span>
									<span className="ml-2">{selectedMatch.num_matches}</span>
								</div>
							)}
							{(() => {
								const duration = getProcessingDuration(selectedMatch);
								return (
									duration && (
										<div>
											<span className="text-muted-foreground">Processing Time:</span>
											<span className="ml-2">{duration.toFixed(1)}s</span>
										</div>
									)
								);
							})()}
							{(() => {
								const metrics = getTranscodeMetrics(selectedMatch);
								return (
									metrics.fps && (
										<div>
											<span className="text-muted-foreground">Transcode FPS:</span>
											<span className="ml-2">{metrics.fps.toFixed(1)} fps</span>
										</div>
									)
								);
							})()}
						</div>

						{/* Quality Settings */}
						{(() => {
							const qualitySettings = getQualitySettings(selectedMatch);
							return (
								qualitySettings && (
									<div className="border-t pt-3">
										<h4 className="text-xs font-semibold mb-2">Processing Quality</h4>
										<div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
											<div>
												<span className="text-muted-foreground">Preset:</span>
												<span className="ml-2 capitalize">{qualitySettings.preset}</span>
											</div>
											{qualitySettings.resolution && (
												<div>
													<span className="text-muted-foreground">Resolution:</span>
													<span className="ml-2">{qualitySettings.resolution}</span>
												</div>
											)}
											{qualitySettings.bitrate && (
												<div>
													<span className="text-muted-foreground">Bitrate:</span>
													<span className="ml-2">{qualitySettings.bitrate}</span>
												</div>
											)}
											{qualitySettings.speed_preset && (
												<div>
													<span className="text-muted-foreground">Speed:</span>
													<span className="ml-2 capitalize">
														{qualitySettings.speed_preset}
													</span>
												</div>
											)}
											{qualitySettings.use_gpu_decode !== undefined && (
												<div>
													<span className="text-muted-foreground">GPU Decode:</span>
													<span className="ml-2">
														{qualitySettings.use_gpu_decode ? 'Enabled' : 'Disabled'}
													</span>
												</div>
											)}
										</div>
									</div>
								)
							);
						})()}

						{/* Divider */}
						<div className="border-t pt-3">
							<p className="text-xs text-muted-foreground mb-3">
								Adjust your preferred viewing range. Changes are saved automatically.
							</p>
						</div>

						{/* Horizontal Panning */}
						<div>
							<Label htmlFor="yaw-range" className="text-xs font-medium">
								Horizontal Range: {yawRange}°
							</Label>
							<Slider
								id="yaw-range"
								min={30}
								max={180}
								step={5}
								value={[yawRange]}
								onValueChange={(value) => setYawRange(value[0])}
								className="mt-2"
							/>
						</div>

						{/* Vertical Panning */}
						<div>
							<Label htmlFor="pitch-range" className="text-xs font-medium">
								Vertical Range: {pitchRange}°
							</Label>
							<Slider
								id="pitch-range"
								min={5}
								max={60}
								step={5}
								value={[pitchRange]}
								onValueChange={(value) => setPitchRange(value[0])}
								className="mt-2"
							/>
						</div>
					</div>
				)}
			</div>
		</div>
	);
};

export default Viewer;
