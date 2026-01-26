import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useMatch } from '@/features/matches/hooks/useMatches';
import Viewer from '@/features/viewer/components/Viewer';
import { Button } from '@/components/ui/button';
import { ChevronLeft } from 'lucide-react';

export default function MatchViewer() {
	const { id } = useParams();
	const navigate = useNavigate();
	const { match, loading, error } = useMatch(id);

	if (loading) {
		return (
			<div className="w-full h-full flex items-center justify-center">
				<p className="text-muted-foreground">Loading match...</p>
			</div>
		);
	}

	if (error || !match) {
		return (
			<div className="w-full h-full flex flex-col items-center justify-center gap-4">
				<p className="text-red-600">Error loading match: {error || 'Match not found'}</p>
				<Button onClick={() => navigate('/')}>
					<ChevronLeft className="h-4 w-4 mr-2" />
					Back to Matches
				</Button>
			</div>
		);
	}

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6">
			<div className="w-full max-w-6xl">
				<div className="mt-6 w-full flex justify-center">
					<section className="w-full max-w-6xl aspect-video flex flex-col">
						<Button variant="ghost" onClick={() => navigate('/')} className="self-start">
							<ChevronLeft className="h-4 w-4 mr-2" />
							Back to Matches
						</Button>
						<Viewer key={match.id} selectedMatch={match} />
					</section>
				</div>
			</div>
		</div>
	);
}
