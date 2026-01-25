import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useMatchProcessing } from '@/features/matches/hooks/useMatchProcessing';
import { getMatch, processMatchWithFrames, cancelProcessing } from '@/features/matches/api/matches';
import ProcessingStatus from '@/features/matches/components/ProcessingStatus';
import FrameExtractor from '@/features/viewer/components/FrameExtractor';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/toast';
import { useSettings } from '@/hooks/useSettings';

/**
 * ProcessingMatch page - shows processing status for a match
 * Allows resuming from where user left off
 */
export default function ProcessingMatch() {
	const { id: matchId } = useParams();
	const navigate = useNavigate();
	const { showToast } = useToast();
	const { settings } = useSettings();

	const [matchData, setMatchData] = React.useState(null);
	const [loading, setLoading] = React.useState(true);
	const [error, setError] = React.useState(null);
	const [extractingMatch, setExtractingMatch] = React.useState(null);
	const extractionOpenedIds = React.useRef(new Set());

	const processing = useMatchProcessing(matchId, {
		pollInterval: 1000,
		autoPoll: true,
	});

	// Update processing state when status changes
	useEffect(() => {
		const isActive = !!(processing.status && !['ready', 'error'].includes(processing.status.status));

		if (window.electronAPI?.setProcessingState) {
			window.electronAPI.setProcessingState(isActive, 'ProcessingMatch');
		}
	}, [processing.status]);

	// Load match data
	useEffect(() => {
		const loadMatch = async () => {
			try {
				setLoading(true);
				const match = await getMatch(matchId);
				setMatchData(match);
				setError(null);
			} catch (err) {
				console.error('Failed to load match:', err);
				setError(err.message || 'Failed to load match');
			} finally {
				setLoading(false);
			}
		};

		if (matchId) {
			loadMatch();
		}
	}, [matchId]);

	// Auto-complete when processing finishes
	useEffect(() => {
		if (extractingMatch) return;

		if (processing.status && (processing.status.status === 'ready' || processing.status.status === 'complete')) {
			showToast({ message: 'Processing complete!', type: 'success' });
		}

		if (processing.status && processing.status.status === 'error') {
			showToast({ message: 'Error during processing', type: 'error' });
		}
	}, [processing.status, extractingMatch]);

	// Auto-open frame extractor when awaiting frames
	useEffect(() => {
		const openExtractor = async () => {
			if (
				matchId &&
				processing.status &&
				processing.status.processing_step === 'awaiting_frames' &&
				!extractingMatch
			) {
				if (extractionOpenedIds.current.has(matchId)) return;
				extractionOpenedIds.current.add(matchId);

				// Stop polling while extracting frames
				processing.stopPolling();

				// Reload match data to get the video src
				let match = matchData;
				if (!match || !match.src) {
					try {
						match = await getMatch(matchId);
						setMatchData(match);
					} catch (err) {
						console.error('Failed to reload match data:', err);
						return;
					}
				}

				if (match && match.src) {
					const apiBaseUrl = settings.apiBaseUrl;
					const baseUrl = apiBaseUrl.replace('/api', '');
					const videoUrl = `${baseUrl}/${match.src}`;

					setExtractingMatch({
						id: matchId,
						name: match.name,
						videoUrl,
						leftUniforms: match.left_uniforms,
						rightUniforms: match.right_uniforms,
					});
				}
			}
		};

		openExtractor();
	}, [processing.status, matchId, extractingMatch, matchData]);

	const handleFrameExtractionComplete = async ({ leftBlob, rightBlob }) => {
		try {
			if (!extractingMatch) return;
			const mId = extractingMatch.id;
			setExtractingMatch(null);

			showToast({ message: 'Uploading extracted frames...', type: 'info' });
			await processMatchWithFrames(mId, leftBlob, rightBlob);

			showToast({ message: 'Calibration started', type: 'info' });
			processing.startPolling();
		} catch (err) {
			console.error('Failed to send frames:', err);
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

	const handleProcessingComplete = () => {
		try {
			processing.stopPolling();
		} catch (err) {
			console.error('Failed to stop polling:', err);
		}

		// Mark match as viewed
		const viewedMatches = JSON.parse(localStorage.getItem('viewedMatches') || '[]');
		if (!viewedMatches.includes(matchId)) {
			viewedMatches.push(matchId);
			localStorage.setItem('viewedMatches', JSON.stringify(viewedMatches));
		}

		navigate(`/viewer/${matchId}`);
	};

	const handleCancel = async () => {
		let shouldCancel = false;
		if (window.electronAPI?.confirmCancelProcessing) {
			shouldCancel = await window.electronAPI.confirmCancelProcessing();
		} else {
			shouldCancel = confirm(
				'Are you sure you want to cancel processing? This will stop the current transcoding operation.'
			);
		}

		if (!shouldCancel) return;

		try {
			processing.stopPolling();
		} catch (err) {
			console.error('Failed to stop polling:', err);
		}

		if (processing.status) {
			const activeStatuses = ['transcoding', 'calibrating'];
			if (activeStatuses.includes(processing.status.status)) {
				try {
					await cancelProcessing(matchId);
					showToast({ message: 'Processing cancelled', type: 'info' });
				} catch (err) {
					console.error('Failed to cancel processing:', err);
					showToast({ message: 'Failed to cancel processing', type: 'error' });
				}
			}
		}

		navigate('/');
	};

	if (loading) {
		return (
			<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6">
				<div className="w-full max-w-4xl space-y-6">
					<div className="flex items-center justify-between">
						<h1 className="text-3xl font-bold">Processing: Match</h1>
						<Button variant="outline" onClick={() => navigate('/')}>
							Back to Home
						</Button>
					</div>
					<ProcessingStatus
						status={{
							status: 'transcoding',
							processing_step: 'transcoding',
							processing_message: 'Starting video processing...',
							progress_percent: 0,
						}}
					/>
				</div>
			</div>
		);
	}

	if (error && !processing.status) {
		return (
			<div className="w-full h-full flex items-center justify-center px-6">
				<Alert variant="destructive" className="max-w-lg">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			</div>
		);
	}

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6">
			<div className="w-full max-w-4xl space-y-6">
				<div className="flex items-center justify-between">
					<h1 className="text-3xl font-bold">Processing: {matchData?.name || 'Match'}</h1>
					<Button variant="outline" onClick={() => navigate('/')}>
						Back to Home
					</Button>
				</div>

				{/* Always show progress - either real or optimistic */}
				<ProcessingStatus
					status={
						!processing.status || processing.status.status === 'pending' || !processing.status.status
							? {
									status: 'transcoding',
									processing_step: 'transcoding',
									processing_message: 'Starting video processing...',
									progress_percent: 0,
								}
							: processing.status
					}
					onComplete={handleProcessingComplete}
				/>

				{error && (
					<Alert variant="destructive">
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}

				<div className="flex gap-2">
					{/* Show cancel when FFmpeg is running (transcoding with fps data) */}
					{processing.status?.status === 'transcoding' && processing.status?.fps && (
						<Button variant="ghost" onClick={handleCancel}>
							Cancel
						</Button>
					)}
				</div>
			</div>

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
		</div>
	);
}
