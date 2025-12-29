import React, { useState, useEffect } from 'react';
import { useToast } from '@/components/ui/toast';
import MatchCreationForm from './MatchCreationForm';
import ProcessingStatus from './ProcessingStatus';
import { useMatchMutations } from '../hooks/useMatches';
import { useMatchProcessing } from '../hooks/useMatchProcessing';
import { getMatch, processMatchWithFrames } from '../api/matches';
import FrameExtractor from '@/features/viewer/components/FrameExtractor';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

export default function MatchWizard({ onComplete, onCancel }) {
	const { showToast } = useToast();

	const [createdMatchId, setCreatedMatchId] = useState(null);
	const [createdMatchData, setCreatedMatchData] = useState(null);
	const [showProcessing, setShowProcessing] = useState(false);
	const [extractingMatch, setExtractingMatch] = useState(null);
	const [error, setError] = useState(null);
	const extractionOpenedIds = React.useRef(new Set());

	const { create } = useMatchMutations();
	const processing = useMatchProcessing(createdMatchId, {
		pollInterval: 5000,
		autoPoll: showProcessing && createdMatchId !== null,
	});

	// Auto-complete wizard when processing finishes successfully
	useEffect(() => {
		if (extractingMatch) {
			return;
		}
		if (
			showProcessing &&
			processing.status &&
			(processing.status.status === 'ready' || processing.status.status === 'complete')
		) {
			showToast({ message: 'Processing complete! Opening viewer...', type: 'success' });
			const timer = setTimeout(() => {
				handleProcessingComplete();
			}, 1000);
			return () => clearTimeout(timer);
		}
		if (showProcessing && processing.status && processing.status.status === 'error') {
			showToast({ message: 'Error during processing', type: 'error' });
		}
	}, [processing.status, showProcessing, extractingMatch]);

	// Auto-open frame extractor when backend indicates frames are required
	useEffect(() => {
		if (
			createdMatchId &&
			processing.status &&
			processing.status.processing_step &&
			processing.status.processing_step === 'awaiting_frames' &&
			!extractingMatch
		) {
			// Only open extractor once per match to avoid duplicate extractions
			if (extractionOpenedIds.current.has(createdMatchId)) return;
			extractionOpenedIds.current.add(createdMatchId);
			// Prepare extractor by fetching match data
			(async () => {
				try {
					const match = await getMatch(createdMatchId);
					const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
					const baseUrl = apiBaseUrl.replace('/api', '');
					const videoUrl = `${baseUrl}/${match.src}`;
					setExtractingMatch({
						id: createdMatchId,
						name: match.name,
						videoUrl,
						leftUniforms: match.left_uniforms,
						rightUniforms: match.right_uniforms,
					});
				} catch (err) {
					console.error('Failed to prepare frame extractor:', err);
					showToast({ message: 'Failed to prepare frame extractor', type: 'error' });
					// remove from opened ids so user can retry later
					extractionOpenedIds.current.delete(createdMatchId);
				}
			})();
		}
	}, [processing.status, createdMatchId, extractingMatch]);

	// Save draft to localStorage whenever state changes
	// Handle Escape key to cancel
	useEffect(() => {
		const handleKeyDown = (e) => {
			if (e.key === 'Escape' && !showProcessing) {
				if (confirm('Cancel match creation?')) {
					onCancel();
				}
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [onCancel, showProcessing]);

	const handleFormSubmit = async (formData) => {
		try {
			setError(null);

			// Generate unique ID from timestamp
			const id = `match-${Date.now()}`;

			// Default calibration params - user can adjust later
			const defaultParams = {
				cameraAxisOffset: 0.23,
				intersect: 0.55,
				zRx: 0.0,
				xTy: 0.0,
				xRz: 0.0,
			};

			// Build uniforms from profile data
			const buildUniforms = (profile) => {
				// Validate profile structure
				if (
					!profile.resolution ||
					typeof profile.resolution.width !== 'number' ||
					typeof profile.resolution.height !== 'number'
				) {
					throw new Error(`Profile ${profile.id} has invalid resolution`);
				}

				if (
					!profile.camera_matrix ||
					typeof profile.camera_matrix.fx !== 'number' ||
					typeof profile.camera_matrix.fy !== 'number' ||
					typeof profile.camera_matrix.cx !== 'number' ||
					typeof profile.camera_matrix.cy !== 'number'
				) {
					throw new Error(`Profile ${profile.id} has invalid camera matrix`);
				}

				if (
					!profile.distortion_coeffs ||
					!Array.isArray(profile.distortion_coeffs) ||
					profile.distortion_coeffs.length !== 4 ||
					!profile.distortion_coeffs.every((c) => typeof c === 'number')
				) {
					throw new Error(`Profile ${profile.id} has invalid distortion coefficients`);
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

			const matchPayload = {
				id,
				name: formData.name,
				left_videos: formData.left_videos,
				right_videos: formData.right_videos,
				params: defaultParams,
				left_uniforms: buildUniforms(formData.leftProfile),
				right_uniforms: buildUniforms(formData.rightProfile),
				metadata: {
					left_profile_id: formData.leftProfile.id,
					right_profile_id: formData.rightProfile.id,
				},
			};

			const createdMatch = await create(matchPayload);

			// Store match ID and show processing
			setCreatedMatchId(createdMatch.id);
			setCreatedMatchData(createdMatch);
			setShowProcessing(true);
		} catch (err) {
			setError(err.message || 'Failed to create match');
		}
	};

	// Auto-start processing after match is created and ID is set
	useEffect(() => {
		if (createdMatchId && showProcessing && !processing.status) {
			// Only auto-start if we haven't started processing yet
			const autoStart = async () => {
				try {
					await processing.startProcessing();
					processing.startPolling();
				} catch (err) {
					console.error('Failed to auto-start processing:', err);
					setError(err.message || 'Failed to start processing');
				}
			};
			autoStart();
		}
	}, [createdMatchId, showProcessing]);

	const handleStartProcessing = async () => {
		try {
			setError(null);
			setShowProcessing(true);

			// If backend already indicates frames are ready, don't re-trigger transcode
			if (processing.status && processing.status.processing_step === 'awaiting_frames') {
				try {
					processing.startPolling();
				} catch {
					// Ignore if startPolling unavailable or already running
				}
				return null;
			}

			const res = await processing.startProcessing();
			try {
				processing.startPolling();
			} catch {
				// Ignore if startPolling unavailable or already running
			}
			return res;
		} catch (err) {
			setError(err.message || 'Failed to start processing');
			throw err;
		}
	};

	const handleOpenExtractor = async () => {
		if (!createdMatchId) return;
		if (extractingMatch) return;
		try {
			setError(null);
			const match = await getMatch(createdMatchId);
			if (!match.src || !match.left_uniforms || !match.right_uniforms) {
				throw new Error('Match missing video source or uniforms');
			}
			const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
			const baseUrl = apiBaseUrl.replace('/api', '');
			const videoUrl = `${baseUrl}/${match.src}`;
			setExtractingMatch({
				id: createdMatchId,
				name: match.name,
				videoUrl,
				leftUniforms: match.left_uniforms,
				rightUniforms: match.right_uniforms,
			});
			// mark as opened to avoid auto-open race
			extractionOpenedIds.current.add(createdMatchId);
		} catch (err) {
			console.error('Failed to open extractor manually:', err);
			showToast({ message: err.message || 'Failed to open extractor', type: 'error' });
			setError(err.message || 'Failed to open extractor');
		}
	};

	// Keep a fresh copy of the created match so the UI can decide whether
	// to show the manual extractor control (only when a video src exists).
	useEffect(() => {
		let mounted = true;
		if (!createdMatchId) {
			setCreatedMatchData(null);
			return;
		}
		(async () => {
			try {
				const m = await getMatch(createdMatchId);
				if (mounted) setCreatedMatchData(m);
			} catch (e) {
				console.warn('Failed to fetch created match data:', e);
			}
		})();
		return () => {
			mounted = false;
		};
	}, [createdMatchId]);

	const handleFrameExtractionComplete = async ({ leftBlob, rightBlob }) => {
		try {
			if (!extractingMatch) return;
			const matchId = extractingMatch.id;
			setExtractingMatch(null);
			// mark that we've sent frames for this match so extractor won't reopen
			try {
				extractionOpenedIds.current.add(matchId);
			} catch {
				// Ignore if ref operations fail
			}
			showToast({ message: 'Uploading extracted frames...', type: 'info' });
			await processMatchWithFrames(matchId, leftBlob, rightBlob);
			// Ensure processing is shown and polling starts
			setShowProcessing(true);
			try {
				processing.startPolling();
			} catch {
				// Ignore if startPolling unavailable or already running
			}
			showToast({ message: 'Calibration started', type: 'info' });
		} catch (err) {
			console.error('Failed to send frames for processing:', err);
			showToast({ message: 'Failed to upload frames', type: 'error' });
			setError(err.message || 'Failed to upload frames');
		}
	};

	const handleFrameExtractionError = (err) => {
		console.error('Frame extraction error:', err);
		setExtractingMatch(null);
		setError(err.message || 'Frame extraction failed');
		showToast({ message: 'Frame extraction failed', type: 'error' });
	};

	const handleProcessingComplete = async () => {
		try {
			// Stop polling before final fetch
			try {
				processing.stopPolling();
			} catch {
				// Ignore if stopPolling unavailable
			}
			// Fetch the processed match data
			const match = await getMatch(createdMatchId);
			// Complete wizard with processed match
			onComplete(match);
		} catch (err) {
			console.error('Failed to fetch match:', err);
			// Fallback: pass just the ID with ready status
			onComplete({ id: createdMatchId, status: 'ready' });
		}
	};

	const handleCancel = () => {
		onCancel();
	};

	const isForegroundProcessing =
		processing.status &&
		// Show overlay when calibrating or when extracting but extractor not open
		(processing.status.status === 'calibrating' || (processing.status.status === 'extracting' && !extractingMatch));

	return (
		<div className="w-full max-w-6xl space-y-6 relative">
			{!showProcessing ? (
				<MatchCreationForm onSubmit={handleFormSubmit} onCancel={handleCancel} />
			) : (
				<div className="space-y-4">
					<h2 className="text-2xl font-bold">Processing Match</h2>

					{/* Processing Status */}
					{processing.status && <ProcessingStatus status={processing.status} />}

					{/* Action Buttons */}
					<div className="flex gap-2">
						{!processing.status || processing.status.status === 'pending' ? (
							<Button
								onClick={handleStartProcessing}
								disabled={processing.loading || isForegroundProcessing}
							>
								{processing.loading ? 'Starting...' : 'Start Processing'}
							</Button>
						) : processing.status.status === 'error' ? (
							<Button onClick={handleStartProcessing} disabled={isForegroundProcessing}>
								Retry Processing
							</Button>
						) : (
							<Button disabled>Processing...</Button>
						)}

						{/* Manual Open Extractor control */}
						{createdMatchId && !extractingMatch && createdMatchData && createdMatchData.src && (
							<Button
								variant="outline"
								onClick={handleOpenExtractor}
								disabled={processing.loading || isForegroundProcessing}
							>
								Open Extractor
							</Button>
						)}

						{extractingMatch && (
							<Button variant="ghost" disabled>
								Extractor Open
							</Button>
						)}

						<Button variant="ghost" onClick={handleCancel} disabled={isForegroundProcessing}>
							Cancel
						</Button>
					</div>

					{/* Foreground modal for extracting/calibrating */}
					{isForegroundProcessing && (
						<Dialog open>
							<div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center">
								<div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-8 flex flex-col items-center gap-4 min-w-[320px]">
									<div className="animate-spin text-blue-500 mb-2">
										<svg width="32" height="32" fill="none" viewBox="0 0 24 24">
											<circle
												cx="12"
												cy="12"
												r="10"
												stroke="currentColor"
												strokeWidth="4"
												opacity="0.2"
											/>
											<path
												d="M22 12a10 10 0 0 1-10 10"
												stroke="currentColor"
												strokeWidth="4"
												strokeLinecap="round"
											/>
										</svg>
									</div>
									<div className="text-lg font-semibold">
										{processing.status.status === 'extracting'
											? 'Extracting frames...'
											: 'Calibrating...'}
									</div>
									<div className="text-muted-foreground text-sm">
										Please wait, this may take a minute.
									</div>
								</div>
							</div>
						</Dialog>
					)}
				</div>
			)}

			{/* Frame extractor modal (foreground) */}
			{extractingMatch && (
				<FrameExtractor
					videoSrc={extractingMatch.videoUrl}
					frameTime={100 / 30}
					leftUniforms={extractingMatch.leftUniforms}
					rightUniforms={extractingMatch.rightUniforms}
					onComplete={handleFrameExtractionComplete}
					onError={handleFrameExtractionError}
				/>
			)}

			{/* Error Alert */}
			{error && (
				<Alert variant="destructive">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			)}
		</div>
	);
}
