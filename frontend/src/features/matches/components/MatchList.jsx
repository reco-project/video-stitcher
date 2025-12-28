import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useMatches, useMatchMutations } from '../hooks/useMatches';
import MatchCard from './MatchCard';
import { Plus, Trash2 } from 'lucide-react';

export default function MatchList({ onSelectMatch, onCreateNew }) {
	const { matches, loading, error, refetch } = useMatches();
	const { delete: deleteMatch } = useMatchMutations();
	const [deleteError, setDeleteError] = React.useState(null);
	const [deletingId, setDeletingId] = React.useState(null);
	const [optimisticDeletes, setOptimisticDeletes] = React.useState(new Set());

	// Filter out optimistically deleted matches
	const displayMatches = matches.filter((m) => !optimisticDeletes.has(m.id));

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
		<Card className="w-full max-w-4xl">
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
							const isReady = match.status === 'ready';
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
											<Button
												onClick={() => onSelectMatch(match)}
												disabled={!isReady}
												variant={isReady ? 'default' : 'outline'}
												size="sm"
												className="flex-1"
											>
												{isReady ? 'View' : 'Incomplete'}
											</Button>
											<Button
												onClick={(e) => {
													e.stopPropagation();
													handleDelete(match.id, match.name || 'Untitled');
												}}
												disabled={isDeleting}
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
