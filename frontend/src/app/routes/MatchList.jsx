import React from 'react';
import { useNavigate } from 'react-router-dom';
import MatchListComponent from '@/features/matches/components/MatchList';
import { processMatch } from '@/features/matches/api/matches';

export default function MatchList() {
	const navigate = useNavigate();

	const handleSelectMatch = (match) => {
		navigate(`/viewer/${match.id}`);
	};

	const handleCreateNew = () => {
		navigate('/create');
	};

	const handleEditMatch = (match) => {
		navigate(`/edit/${match.id}`);
	};

	const handleResumeProcessing = async (match) => {
		// If awaiting frames or already processing, just navigate - don't restart
		if (match.processing_step === 'awaiting_frames' || ['transcoding', 'calibrating'].includes(match.status)) {
			navigate(`/processing/${match.id}`);
			return;
		}

		// Only start processing if pending (not yet started)
		try {
			await processMatch(match.id);
		} catch (err) {
			console.error('Failed to start processing:', err);
			// Navigate anyway so user can see error and retry
		}
		navigate(`/processing/${match.id}`);
	};

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6 pb-12">
			<div className="w-full max-w-6xl">
				<div className="mt-6 flex justify-center">
					<MatchListComponent
						onSelectMatch={handleSelectMatch}
						onCreateNew={handleCreateNew}
						onResumeProcessing={handleResumeProcessing}
						onEditMatch={handleEditMatch}
					/>
				</div>
			</div>
		</div>
	);
}
