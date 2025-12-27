import React, { useState } from 'react';
import VideoImportStep from './VideoImportStep';
import ProfileAssignmentStep from './ProfileAssignmentStep';
import { useMatchMutations } from '../hooks/useMatches';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function MatchWizard({ onComplete, onCancel }) {
	const [step, setStep] = useState(1);
	const [matchData, setMatchData] = useState({
		name: '',
		left_videos: [{ path: '', profile_id: null }],
		right_videos: [{ path: '', profile_id: null }],
	});
	const [error, setError] = useState(null);

	const { create } = useMatchMutations();

	const handleStep1Complete = (data) => {
		setMatchData(data);
		setStep(2);
		setError(null);
	};

	const handleStep2Complete = async (finalData) => {
		try {
			setError(null);

			// Generate unique ID from timestamp
			const id = `match-${Date.now()}`;

			const matchPayload = {
				id,
				name: finalData.name,
				left_videos: finalData.left_videos,
				right_videos: finalData.right_videos,
				metadata: {},
			};

			const createdMatch = await create(matchPayload);
			onComplete(createdMatch);
		} catch (err) {
			setError(err.message || 'Failed to create match');
		}
	};

	const handleBack = () => {
		setStep(1);
		setError(null);
	};

	return (
		<div className="w-full max-w-4xl">
			{error && (
				<Alert variant="destructive" className="mb-4">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			)}

			{step === 1 && <VideoImportStep onNext={handleStep1Complete} initialData={matchData} />}

			{step === 2 && (
				<ProfileAssignmentStep matchData={matchData} onNext={handleStep2Complete} onBack={handleBack} />
			)}
		</div>
	);
}
