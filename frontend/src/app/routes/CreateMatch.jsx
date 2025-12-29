import React from 'react';
import { useNavigate } from 'react-router-dom';
import MatchWizard from '@/features/matches/components/MatchWizard';

export default function CreateMatch() {
	const navigate = useNavigate();

	const handleWizardComplete = async (newMatch) => {
		// Auto-navigate to viewer if match is ready, otherwise show list
		if (newMatch.status === 'ready' && newMatch.params) {
			navigate(`/viewer/${newMatch.id}`);
		} else {
			// Show match list for incomplete matches
			navigate('/');
		}
	};

	const handleCancel = () => {
		navigate('/');
	};

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6">
			<div className="w-full max-w-6xl">
				<div className="mt-6 flex justify-center">
					<MatchWizard onComplete={handleWizardComplete} onCancel={handleCancel} />
				</div>
			</div>
		</div>
	);
}
