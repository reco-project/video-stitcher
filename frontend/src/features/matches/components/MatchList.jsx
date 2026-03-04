import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useMatches, useMatchMutations } from '../hooks/useMatches';
import { Pencil, Plus, Trash2, Play, RotateCcw } from 'lucide-react';
import MatchCard from './MatchCard';

export default function MatchList({ onSelectMatch, onCreateNew, onResumeProcessing, onEditMatch }) {
	const { matches, loading, error, refetch } = useMatches();
	const { delete: deleteMatch, create: createMatch, update: updateMatch } = useMatchMutations();
	const [deleteError, setDeleteError] = React.useState(null);
	const [deletingId, setDeletingId] = React.useState(null);
	const [optimisticDeletes, setOptimisticDeletes] = React.useState(new Set());
	const [liveEnsured, setLiveEnsured] = React.useState(false);
	const liveSrc = 'videos/live/index.m3u8';

	// Filter out optimistically deleted matches and sort by most recent (ID descending)
	const visibleMatches = matches.filter((m) => !optimisticDeletes.has(m.id));
	const liveMatch = visibleMatches.find((m) => m.id === 'live');
	const sortedMatches = visibleMatches.filter((m) => m.id !== 'live').sort((a, b) => b.id.localeCompare(a.id));
	const displayMatches = liveMatch ? [liveMatch, ...sortedMatches] : sortedMatches;

	React.useEffect(() => {
		if (loading || liveEnsured) return;

		const hasLive = matches.some((m) => m.id === 'live');
		if (hasLive) {
			const existingLive = matches.find((m) => m.id === 'live');
			if (existingLive && !existingLive.src) {
				updateMatch('live', { id: 'live', src: liveSrc })
					.then(() => refetch())
					.catch(() => {})
					.finally(() => setLiveEnsured(true));
				return;
			}
			setLiveEnsured(true);
			return;
		}

		const baseMatch = [...matches]
			.filter((m) => m.status === 'ready' || m.status === 'warning')
			.sort((a, b) => b.id.localeCompare(a.id))[0];

		const defaultParams = baseMatch?.params || {
			cameraAxisOffset: 0.7,
			intersect: 0.5,
			xTy: 0.0,
			xRz: 0.0,
			zRx: 0.0,
		};

		const leftProfileId = baseMatch?.left_videos?.[0]?.profile_id || baseMatch?.metadata?.left_profile_id || null;
		const rightProfileId =
			baseMatch?.right_videos?.[0]?.profile_id || baseMatch?.metadata?.right_profile_id || null;

		const livePayload = {
			id: 'live',
			name: 'Live',
			src: liveSrc,
			left_videos: [],
			right_videos: [],
			left_uniforms: baseMatch?.left_uniforms || null,
			right_uniforms: baseMatch?.right_uniforms || null,
			params: defaultParams,
			metadata: {
				...(baseMatch?.metadata || {}),
				is_live: true,
				left_profile_id: leftProfileId,
				right_profile_id: rightProfileId,
			},
			processing: {
				status: 'ready',
				step: 'complete',
				message: 'Live stream match',
				error_message: null,
				error_code: null,
			},
		};

		createMatch(livePayload)
			.then(() => refetch())
			.catch(() => {})
			.finally(() => setLiveEnsured(true));
	}, [loading, liveEnsured, matches, createMatch, refetch]);

	// Clear optimistic deletes when matches update (after refetch)
	React.useEffect(() => {
		setOptimisticDeletes(new Set());
	}, [matches]);

	const handleDelete = async (matchId, matchName) => {
		if (!confirm(`Are you sure you want to delete "${matchName}"?`)) {
			return;
		}

		try {
			setDeleteError(null);
			setDeletingId(matchId);

			// Optimistically remove from UI
			setOptimisticDeletes((prev) => new Set([...prev, matchId]));

			// Delete from backend
			await deleteMatch(matchId);

			// Refetch to sync with backend state
			await refetch();
		} catch (err) {
			setDeleteError(err.message || 'Failed to delete match');
			// Revert optimistic update on error
			setOptimisticDeletes((prev) => {
				const next = new Set(prev);
				next.delete(matchId);
				return next;
			});
		} finally {
			setDeletingId(null);
		}
	};

	if (loading) {
		return <div className="text-center p-4">Loading matches...</div>;
	}

	return (
		<Card className="w-full max-w-4xl mb-12">
			<CardHeader>
				<div className="flex justify-between items-center">
					<CardTitle>Saved Matches</CardTitle>
					<Button onClick={onCreateNew} className="gap-2">
						<Plus className="h-4 w-4" />
						Create New Match
					</Button>
				</div>
			</CardHeader>
			<CardContent>
				{(error || deleteError) && (
					<Alert variant="destructive" className="mb-4">
						<AlertDescription>{error || deleteError}</AlertDescription>
					</Alert>
				)}

				{displayMatches.length === 0 ? (
					<div className="text-center py-12 text-muted-foreground">
						<p className="mb-2 text-lg">No matches yet</p>
						<p className="text-sm mb-6">Create your first match to get started</p>
						<Button onClick={onCreateNew} size="lg">
							<Plus className="h-4 w-4 mr-2" />
							Create Your First Match
						</Button>
					</div>
				) : (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{displayMatches.map((match) => {
							// Check if match is complete by looking at both status and required fields
							// This handles legacy matches that have data but status wasn't updated
							const hasRequiredData =
								match.src && match.params && match.left_uniforms && match.right_uniforms;
							// Match is ready if status says so, OR if it has all data AND isn't awaiting frames
							const isReady =
								match.status === 'ready' ||
								match.status === 'warning' ||
								(hasRequiredData && match.processing_step !== 'awaiting_frames');
							const isCancelled = match.processing_message?.toLowerCase().includes('cancelled');
							const isError = match.status === 'error' || isCancelled;
							const isProcessing = ['transcoding', 'calibrating'].includes(match.status);
							const isAwaitingFrames =
								match.status === 'pending' &&
								match.processing_step === 'awaiting_frames' &&
								!isCancelled &&
								!hasRequiredData; // Don't show awaiting frames if match is actually complete
							// Match is ready to process if pending with no step and no src (newly created)
							const isReadyToProcess =
								match.status === 'pending' && !match.processing_step && !match.src && !isCancelled;
							const isDeleting = deletingId === match.id;

							return (
								<div
									key={match.id}
									className="border rounded-lg overflow-hidden hover:border-primary/50 hover:shadow-md transition-all"
								>
									<div className="p-4 h-full flex flex-col gap-3">
										{/* Match preview card with thumbnail */}
										<div onClick={() => isReady && onSelectMatch(match)} className="cursor-pointer">
											<MatchCard match={match} />
										</div>

										{/* Action buttons */}
										<div className="flex gap-2 border-t pt-3">
											{isReady ? (
												<Button
													onClick={() => onSelectMatch(match)}
													variant="default"
													size="sm"
													className="flex-1"
												>
													View
												</Button>
											) : isError ? (
												<>
													<Button
														onClick={() => onResumeProcessing && onResumeProcessing(match)}
														variant="destructive"
														size="sm"
														className="flex-1 gap-2"
													>
														<RotateCcw className="h-3 w-3" />
														Retry
													</Button>
													<Button
														onClick={() => onEditMatch && onEditMatch(match)}
														variant="outline"
														size="sm"
														className="flex-1 gap-2"
													>
														<Pencil className="h-3 w-3" />
														Edit
													</Button>
												</>
											) : isProcessing ? (
												<Button
													onClick={() => onResumeProcessing && onResumeProcessing(match)}
													variant="default"
													size="sm"
													className="flex-1 gap-2"
												>
													<Play className="h-3 w-3" />
													See Progress
												</Button>
											) : isAwaitingFrames ? (
												<Button
													onClick={() => onResumeProcessing && onResumeProcessing(match)}
													variant="default"
													size="sm"
													className="flex-1"
												>
													Calibrate
												</Button>
											) : isReadyToProcess ? (
												<>
													<Button
														onClick={() => onResumeProcessing && onResumeProcessing(match)}
														variant="default"
														size="sm"
														className="flex-1 gap-2"
													>
														<Play className="h-3 w-3" />
														Process
													</Button>
													<Button
														onClick={() => onEditMatch && onEditMatch(match)}
														variant="outline"
														size="sm"
														className="flex-1 gap-2"
													>
														<Pencil className="h-3 w-3" />
														Edit
													</Button>
												</>
											) : (
												<Button
													onClick={() => onSelectMatch(match)}
													disabled
													variant="outline"
													size="sm"
													className="flex-1"
												>
													Incomplete
												</Button>
											)}
											<Button
												onClick={(e) => {
													e.stopPropagation();
													if (match.id === 'live') return;
													handleDelete(match.id, match.name || 'Untitled');
												}}
												disabled={isDeleting || match.id === 'live'}
												variant="ghost"
												size="sm"
												className="text-red-600 hover:text-red-700 hover:bg-red-50"
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										</div>
									</div>
								</div>
							);
						})}
					</div>
				)}
			</CardContent>
		</Card>
	);
}
