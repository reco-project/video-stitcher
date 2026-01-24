import React from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useMatch } from '@/features/matches/hooks/useMatches';
import MatchWizard from '@/features/matches/components/MatchWizard';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function EditMatch() {
	const { id } = useParams();
	const navigate = useNavigate();
	const { match, loading, error } = useMatch(id);

	const handleComplete = (updatedMatch, startProcessing) => {
		// Navigate to processing page if user clicked Save & Process, otherwise home
		if (startProcessing) {
			navigate(`/processing/${updatedMatch.id}`);
		} else {
			navigate('/');
		}
	};

	const handleCancel = () => {
		navigate('/');
	};

	if (loading) {
		return (
			<div className="w-full h-full flex items-center justify-center">
				<div className="text-center">
					<div className="text-lg font-medium">Loading match...</div>
				</div>
			</div>
		);
	}

	if (error) {
		return (
			<div className="w-full h-full flex items-center justify-center p-6">
				<Alert variant="destructive" className="max-w-md">
					<AlertDescription>Failed to load match: {error.message || 'Unknown error'}</AlertDescription>
				</Alert>
			</div>
		);
	}

	if (!match) {
		return (
			<div className="w-full h-full flex items-center justify-center p-6">
				<Alert variant="destructive" className="max-w-md">
					<AlertDescription>Match not found</AlertDescription>
				</Alert>
			</div>
		);
	}

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6 pb-12 overflow-y-auto">
			<div className="w-full max-w-6xl">
				<h1 className="text-2xl font-bold mb-6">Edit Match</h1>
				<MatchWizard onComplete={handleComplete} onCancel={handleCancel} initialMatch={match} />
			</div>
		</div>
	);
}
