import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useMatches, useMatchMutations } from '../hooks/useMatches';
import { getMatchStatus, transcodeMatch, processMatchWithFrames, getMatch } from '../api/matches';
import ProcessingStatus from './ProcessingStatus';
import FrameExtractor from '@/features/viewer/components/FrameExtractor';
import { Loader2, CheckCircle, XCircle, Clock, Play } from 'lucide-react';

const getStatusBadge = (status) => {
	switch (status) {
		case 'pending':
			return (
				<Badge variant="secondary" className="gap-1">
					<Clock className="h-3 w-3" />
					Pending
				</Badge>
			);
		case 'transcoding':
			return (
				<Badge variant="default" className="gap-1 bg-blue-500">
					<Loader2 className="h-3 w-3 animate-spin" />
					Syncing
				</Badge>
			);
		case 'calibrating':
			return (
				<Badge variant="default" className="gap-1 bg-blue-500">
					<Loader2 className="h-3 w-3 animate-spin" />
					Calibrating
				</Badge>
			);
		case 'ready':
			return (
				<Badge variant="default" className="gap-1 bg-green-500">
					<CheckCircle className="h-3 w-3" />
					Ready
				</Badge>
			);
		case 'error':
			return (
				<Badge variant="destructive" className="gap-1">
					<XCircle className="h-3 w-3" />
					Error
				</Badge>
			);
		default:
			return null;
	}
};

export default function MatchList({ onSelectMatch, onCreateNew }) {
	const { matches, loading, error, refetch } = useMatches();
	const { delete: deleteMatch } = useMatchMutations();
	const [deleteError, setDeleteError] = React.useState(null);
	const [deletingId, setDeletingId] = React.useState(null);
	const [processingId, setProcessingId] = React.useState(null);
	const [showProcessingDialog, setShowProcessingDialog] = React.useState(false);
	const [processingMatchName, setProcessingMatchName] = React.useState('');
	const [errorDialogMatch, setErrorDialogMatch] = React.useState(null);
	const [processingStatus, setProcessingStatus] = React.useState(null);
	const [extractingMatch, setExtractingMatch] = React.useState(null);
	const pollIntervalRef = React.useRef(null);

	// Poll for status updates
	const startPolling = React.useCallback(
		(matchId) => {
			// Stop any existing polling
			if (pollIntervalRef.current) {
				clearInterval(pollIntervalRef.current);
			}

			// Fetch status immediately
			const fetchStatus = async () => {
				try {
					const status = await getMatchStatus(matchId);
					setProcessingStatus(status);

					// Stop polling if complete or errored
					if (status.status === 'ready' || status.status === 'error') {
						if (pollIntervalRef.current) {
							clearInterval(pollIntervalRef.current);
							pollIntervalRef.current = null;
						}
						refetch(); // Refresh the match list
					}
				} catch (err) {
					console.error('Failed to fetch status:', err);
				}
			};

			fetchStatus(); // Initial fetch

			// Poll every 1 second for faster updates
			pollIntervalRef.current = setInterval(fetchStatus, 1000);
		},
		[refetch]
	);

	const stopPolling = React.useCallback(() => {
		if (pollIntervalRef.current) {
			clearInterval(pollIntervalRef.current);
			pollIntervalRef.current = null;
		}
	}, []);

	// Cleanup on unmount
	React.useEffect(() => {
		return () => stopPolling();
	}, [stopPolling]);

	const handleStartProcessing = async (matchId, matchName, skipTranscode = false) => {
		// New two-step processing: transcode then warp frames
		try {
			setProcessingId(matchId);
			setProcessingMatchName(matchName);
			setShowProcessingDialog(true);
			setProcessingStatus(null);

			// Get match data to access uniforms
			const match = await getMatch(matchId);

			// If skipTranscode is true and video already exists, go straight to frame extraction
			if (skipTranscode && match.src) {
				console.log('Skipping transcode, going directly to frame extraction');
				await extractAndSendFrames(matchId, match);
				return;
			}

			// Step 1: Transcode video
			await transcodeMatch(matchId);

			// Poll until transcoding complete
			const pollTranscode = async () => {
				const status = await getMatchStatus(matchId);
				setProcessingStatus(status);

				if (status.status === 'error') {
					clearInterval(pollIntervalRef.current);
					return;
				}

				if (status.processing_step === 'awaiting_frames') {
					// Transcoding done, now extract and warp frames
					clearInterval(pollIntervalRef.current);
					const updatedMatch = await getMatch(matchId);
					await extractAndSendFrames(matchId, updatedMatch);
				}
			};

			pollTranscode();
			pollIntervalRef.current = setInterval(pollTranscode, 1000);
		} catch (err) {
			console.error('Failed to start processing:', err);
			setProcessingId(null);
			setShowProcessingDialog(false);
		}
	};

	const extractAndSendFrames = async (matchId, match) => {
		try {
			if (!match.src) {
				throw new Error('Video source not found in match data');
			}

			if (!match.left_uniforms || !match.right_uniforms) {
				throw new Error('Lens uniforms not found in match data');
			}

			// Construct video URL
			const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
			const baseUrl = apiBaseUrl.replace('/api', '');
			const videoUrl = `${baseUrl}/${match.src}`;

			console.log('Preparing to extract frames from:', videoUrl);

			// Show frame extraction UI
			setExtractingMatch({
				id: matchId,
				name: match.name,
				videoUrl,
				leftUniforms: match.left_uniforms,
				rightUniforms: match.right_uniforms,
			});
		} catch (err) {
			console.error('Frame extraction setup failed:', err);
			setProcessingStatus({
				status: 'error',
				error_message: 'Failed to prepare frame extraction: ' + err.message,
			});
		}
	};

	const handleFrameExtractionComplete = async ({ leftBlob, rightBlob }) => {
		try {
			const matchId = extractingMatch.id;

			setExtractingMatch(null);
			console.log('Frames extracted:', leftBlob.size, 'bytes (left),', rightBlob.size, 'bytes (right)');

			// Debug: Download frames to check them
			if (window.DEBUG_SAVE_FRAMES) {
				const leftUrl = URL.createObjectURL(leftBlob);
				const rightUrl = URL.createObjectURL(rightBlob);
				const leftLink = document.createElement('a');
				leftLink.href = leftUrl;
				leftLink.download = 'debug_left_frame.png';
				leftLink.click();
				const rightLink = document.createElement('a');
				rightLink.href = rightUrl;
				rightLink.download = 'debug_right_frame.png';
				rightLink.click();
				URL.revokeObjectURL(leftUrl);
				URL.revokeObjectURL(rightUrl);
			}

			// Send warped frames to backend
			await processMatchWithFrames(matchId, leftBlob, rightBlob);

			// Continue polling for calibration
			startPolling(matchId);
		} catch (err) {
			console.error('Frame processing failed:', err);
			setProcessingStatus({
				status: 'error',
				error_message: 'Failed to process warped frames: ' + err.message,
			});
		}
	};

	const handleFrameExtractionError = (error) => {
		console.error('Frame extraction error:', error);
		setExtractingMatch(null);
		setProcessingStatus({
			status: 'error',
			error_message: 'Frame extraction failed: ' + error.message,
		});
	};

	const handleCloseProcessingDialog = () => {
		stopPolling();
		setShowProcessingDialog(false);
		setProcessingId(null);
		setProcessingMatchName('');
		setProcessingStatus(null);
	};

	const handleDelete = async (matchId, matchName) => {
		if (!confirm(`Are you sure you want to delete "${matchName}"?`)) {
			return;
		}

		try {
			setDeleteError(null);
			setDeletingId(matchId);
			await deleteMatch(matchId);
			await refetch();
		} catch (err) {
			setDeleteError(err.message || 'Failed to delete match');
		} finally {
			setDeletingId(null);
		}
	};

	if (loading) {
		return <div className="text-center p-4">Loading matches...</div>;
	}

	return (
		<>
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

			<Card className="w-full max-w-4xl">
				<CardHeader>
					<div className="flex justify-between items-center">
						<CardTitle>Saved Matches</CardTitle>
						<Button onClick={onCreateNew}>+ Create New Match</Button>
					</div>
				</CardHeader>
				<CardContent>
					{(error || deleteError) && (
						<Alert variant="destructive" className="mb-4">
							<AlertDescription>{error || deleteError}</AlertDescription>
						</Alert>
					)}

					{matches.length === 0 ? (
						<div className="text-center py-8 text-muted-foreground">
							<p className="mb-4">No matches yet</p>
							<Button onClick={onCreateNew} variant="outline">
								Create your first match
							</Button>
						</div>
					) : (
						<div className="space-y-3">
							{matches.map((match) => (
								<div
									key={match.id}
									className="flex items-center justify-between p-4 border rounded hover:bg-accent transition-colors"
								>
									<div className="flex-1">
										<div className="flex items-center gap-2 mb-1">
											<h3 className="font-semibold">{match.name || match.label}</h3>
											{match.status && getStatusBadge(match.status)}
										</div>
										<div className="text-sm text-muted-foreground mt-1">
											{match.src && (
												<span className="truncate max-w-md inline-block">{match.src}</span>
											)}
											{!match.src && match.status === 'pending' && (
												<span className="text-amber-600">Not processed yet</span>
											)}
											{match.created_at && (
												<>
													<span className="mx-2">•</span>
													<span>{new Date(match.created_at).toLocaleDateString()}</span>
												</>
											)}
										</div>
										{match.error_message && (
											<div className="text-xs text-red-500 mt-1 flex items-center gap-2">
												<span className="truncate max-w-md">
													{match.error_message.split('\n')[0]}
												</span>
												<Button
													variant="ghost"
													size="sm"
													className="h-5 px-2 text-xs"
													onClick={() => setErrorDialogMatch(match)}
												>
													View Details
												</Button>
											</div>
										)}
									</div>
									<div className="flex gap-2">
										{/* Show Process button for pending matches without video */}
										{match.status === 'pending' && !match.src && (
											<Button
												onClick={() =>
													handleStartProcessing(match.id, match.name || match.label, false)
												}
												variant="default"
												size="sm"
												disabled={processingId === match.id}
												className="gap-1"
											>
												{processingId === match.id ? (
													<>
														<Loader2 className="h-3 w-3 animate-spin" />
														Processing...
													</>
												) : (
													<>
														<Play className="h-3 w-3" />
														Process Now
													</>
												)}
											</Button>
										)}

										{/* Show Continue button for pending matches with video (awaiting_frames) */}
										{match.status === 'pending' &&
											match.src &&
											match.processing_step === 'awaiting_frames' && (
												<>
													<Button
														onClick={() =>
															handleStartProcessing(
																match.id,
																match.name || match.label,
																true
															)
														}
														variant="default"
														size="sm"
														disabled={processingId === match.id}
														className="gap-1"
													>
														{processingId === match.id ? (
															<>
																<Loader2 className="h-3 w-3 animate-spin" />
																Processing...
															</>
														) : (
															<>
																<Play className="h-3 w-3" />
																Continue Processing
															</>
														)}
													</Button>
													<Button
														onClick={() =>
															handleStartProcessing(
																match.id,
																match.name || match.label,
																false
															)
														}
														variant="outline"
														size="sm"
														disabled={processingId === match.id}
														className="gap-1"
													>
														Start Over
													</Button>
												</>
											)}

										{/* Show Retry button for failed matches */}
										{match.status === 'error' && (
											<>
												<Button
													onClick={() =>
														handleStartProcessing(
															match.id,
															match.name || match.label,
															false
														)
													}
													variant="default"
													size="sm"
													disabled={processingId === match.id}
													className="gap-1"
												>
													{processingId === match.id ? (
														<>
															<Loader2 className="h-3 w-3 animate-spin" />
															Retrying...
														</>
													) : (
														<>
															<Play className="h-3 w-3" />
															Retry From Start
														</>
													)}
												</Button>
												{/* If video exists, allow continuing from frame extraction */}
												{match.src && (
													<Button
														onClick={() =>
															handleStartProcessing(
																match.id,
																match.name || match.label,
																true
															)
														}
														variant="outline"
														size="sm"
														disabled={processingId === match.id}
														className="gap-1"
													>
														Continue from Frames
													</Button>
												)}
											</>
										)}

										{/* Show Reprocess button for ready matches */}
										{match.status === 'ready' && (
											<>
												<Button
													onClick={() =>
														handleStartProcessing(match.id, match.name || match.label, true)
													}
													variant="default"
													size="sm"
													disabled={processingId === match.id}
													className="gap-1"
												>
													{processingId === match.id ? (
														<>
															<Loader2 className="h-3 w-3 animate-spin" />
															Recalibrating...
														</>
													) : (
														<>
															<Play className="h-3 w-3" />
															Recalibrate
														</>
													)}
												</Button>
												<Button
													onClick={() =>
														handleStartProcessing(
															match.id,
															match.name || match.label,
															false
														)
													}
													variant="outline"
													size="sm"
													disabled={processingId === match.id}
													className="gap-1"
												>
													{processingId === match.id ? (
														<>
															<Loader2 className="h-3 w-3 animate-spin" />
															Reprocessing...
														</>
													) : (
														<>
															<Play className="h-3 w-3" />
															Reprocess All
														</>
													)}
												</Button>
											</>
										)}

										{/* Show View button for ready matches, disabled for processing */}
										<Button
											onClick={() => onSelectMatch(match)}
											variant="outline"
											size="sm"
											disabled={match.status !== 'ready' && match.status !== undefined}
										>
											{match.status === 'ready' ? 'View' : 'View Details'}
										</Button>

										<Button
											onClick={() => handleDelete(match.id, match.name || match.label)}
											variant="destructive"
											size="sm"
											disabled={deletingId === match.id}
										>
											{deletingId === match.id ? 'Deleting...' : 'Delete'}
										</Button>
									</div>
								</div>
							))}
						</div>
					)}
				</CardContent>
			</Card>

			{/* Processing Dialog */}
			<Dialog
				open={showProcessingDialog}
				onOpenChange={(open) => {
					if (!open) {
						handleCloseProcessingDialog();
					}
				}}
			>
				<DialogContent className="max-w-2xl">
					<DialogHeader>
						<DialogTitle>Processing Match: {processingMatchName}</DialogTitle>
						<DialogDescription>
							Transcoding videos and calibrating camera parameters for stitching
						</DialogDescription>
					</DialogHeader>

					<div className="py-4">
						{processingStatus ? (
							<ProcessingStatus status={processingStatus} />
						) : (
							<div className="flex items-center justify-center p-8">
								<Loader2 className="h-8 w-8 animate-spin text-primary" />
								<span className="ml-2">Starting processing...</span>
							</div>
						)}
					</div>

					<div className="flex justify-end gap-2">
						{processingStatus?.status === 'ready' && (
							<Button onClick={handleCloseProcessingDialog} variant="default">
								Done
							</Button>
						)}
						{processingStatus?.status === 'error' && (
							<Button onClick={handleCloseProcessingDialog} variant="outline">
								Close
							</Button>
						)}
						{processingStatus?.status !== 'ready' && processingStatus?.status !== 'error' && (
							<Button onClick={handleCloseProcessingDialog} variant="outline">
								Close (runs in background)
							</Button>
						)}
					</div>
				</DialogContent>
			</Dialog>

			{/* Error Details Dialog */}
			<Dialog open={!!errorDialogMatch} onOpenChange={(open) => !open && setErrorDialogMatch(null)}>
				<DialogContent className="max-w-3xl">
					<DialogHeader>
						<DialogTitle>Error Details: {errorDialogMatch?.name || errorDialogMatch?.label}</DialogTitle>
						<DialogDescription>Complete error information for this match</DialogDescription>
					</DialogHeader>

					<div className="space-y-4">
						{errorDialogMatch?.error_code && (
							<div>
								<div className="text-sm font-medium mb-1">Error Code</div>
								<div className="p-2 bg-muted rounded text-sm font-mono">
									{errorDialogMatch.error_code}
								</div>
							</div>
						)}

						<div>
							<div className="text-sm font-medium mb-1">Error Message</div>
							<div className="p-3 bg-muted rounded text-sm font-mono overflow-x-auto max-h-96 overflow-y-auto">
								<pre className="whitespace-pre-wrap break-words">
									{errorDialogMatch?.error_message || 'No error message available'}
								</pre>
							</div>
						</div>

						{errorDialogMatch?.processing_started_at && (
							<div className="text-xs text-muted-foreground">
								Processing started: {new Date(errorDialogMatch.processing_started_at).toLocaleString()}
							</div>
						)}
						{errorDialogMatch?.processing_completed_at && (
							<div className="text-xs text-muted-foreground">
								Failed at: {new Date(errorDialogMatch.processing_completed_at).toLocaleString()}
							</div>
						)}
					</div>

					<div className="flex justify-end">
						<Button onClick={() => setErrorDialogMatch(null)} variant="outline">
							Close
						</Button>
					</div>
				</DialogContent>
			</Dialog>
		</>
	);
}
