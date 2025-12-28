import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useMatches, useMatchMutations } from '../hooks/useMatches';

export default function MatchList({ onSelectMatch, onCreateNew }) {
	const { matches, loading, error, refetch } = useMatches();
	const { delete: deleteMatch } = useMatchMutations();
	const [deleteError, setDeleteError] = React.useState(null);
	const [deletingId, setDeletingId] = React.useState(null);

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
									<h3 className="font-semibold">{match.name || match.label}</h3>
									<div className="text-sm text-muted-foreground mt-1">
										<span className="truncate max-w-md inline-block">{match.src}</span>
										{match.created_at && (
											<>
												<span className="mx-2">•</span>
												<span>{new Date(match.created_at).toLocaleDateString()}</span>
											</>
										)}
									</div>
								</div>
								<div className="flex gap-2">
									<Button onClick={() => onSelectMatch(match)} variant="outline" size="sm">
										View
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
	);
}
