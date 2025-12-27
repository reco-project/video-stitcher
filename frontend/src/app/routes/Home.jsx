import React, { useState } from 'react';
import Viewer from '@/features/viewer/components/Viewer.jsx';
import Health from '@/features/health/components/Health.jsx';
import { useNavigateTo } from '../Router';
import { Button } from '@/components/ui/button';
import MatchWizard from '@/features/matches/components/MatchWizard';
import { useMatches } from '@/features/matches/hooks/useMatches';
import matches from '@/data/matches.js';

export default function Home() {
	const [selectedMatch, setSelectedMatch] = useState(null);
	const [showWizard, setShowWizard] = useState(false);
	const navigate = useNavigateTo();

	const { matches: sessionMatches, refetch } = useMatches();

	// Combine legacy hardcoded matches with session matches
	const allMatches = [...matches, ...sessionMatches];

	const handleWizardComplete = async (newMatch) => {
		await refetch();
		setShowWizard(false);
		// Auto-select the newly created match
		setSelectedMatch(newMatch);
	};

	return (
		<div className="flex flex-col items-center w-full p-4 gap-4">
			<h1 className="text-purple-600">Video Stitcher</h1>
			<p>Welcome — this is the renderer application root.</p>

			<div className="flex gap-2">
				<Button onClick={() => setShowWizard(true)}>+ Create New Match</Button>
				<Button variant="outline" onClick={navigate.toProfiles}>
					Manage Lens Profiles
				</Button>
			</div>

			<Health />

			{showWizard ? (
				<MatchWizard onComplete={handleWizardComplete} onCancel={() => setShowWizard(false)} />
			) : (
				<>
					<div className="w-full max-w-2xl">
						<label className="block mb-2 font-bold">Select match</label>
						<select
							className="w-full p-2 rounded border"
							value={selectedMatch ? selectedMatch.id : ''}
							onChange={(e) => {
								const id = e.target.value;
								const m = allMatches.find((mm) => mm.id === id) || null;
								setSelectedMatch(m);
							}}
						>
							<option value="">-- choose match --</option>
							{allMatches.map((m) => (
								<option key={m.id} value={m.id}>
									{m.label || m.name}
								</option>
							))}
						</select>
					</div>

					{selectedMatch && (
						<section className={'w-full aspect-video h-full flex flex-col items-center align-middle'}>
							<Viewer key={selectedMatch.id} selectedMatch={selectedMatch} />
						</section>
					)}
				</>
			)}
		</div>
	);
}
