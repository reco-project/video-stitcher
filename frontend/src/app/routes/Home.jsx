import React, { useState, useEffect } from 'react';
import Viewer from '@/features/viewer/components/Viewer.jsx';
import Health from '@/features/health/components/Health.jsx';
import { useNavigateTo } from '../Router';
import { Button } from '@/components/ui/button';
import MatchWizard from '@/features/matches/components/MatchWizard';
import MatchList from '@/features/matches/components/MatchList';
import { useMatchMutations } from '@/features/matches/hooks/useMatches';
import legacyMatches from '@/data/matches.js';

const LEGACY_LOADED_KEY = 'legacyMatchesLoaded';

export default function Home() {
	const [selectedMatch, setSelectedMatch] = useState(null);
	const [showWizard, setShowWizard] = useState(false);
	const [showList, setShowList] = useState(true); // Start with Browse view
	const navigate = useNavigateTo();
	const { create } = useMatchMutations();

	// Load legacy matches into DB once on first run
	useEffect(() => {
		const loadLegacyMatches = async () => {
			const alreadyLoaded = localStorage.getItem(LEGACY_LOADED_KEY);
			if (alreadyLoaded) return;

			try {
				for (const match of legacyMatches) {
					// Convert legacy format: uniforms -> left_uniforms & right_uniforms
					const matchPayload = {
						id: match.id,
						label: match.label,
						name: match.label, // Use label as name
						src: match.src,
						params: match.params,
						left_uniforms: match.uniforms,
						right_uniforms: match.uniforms, // Same uniforms for both sides
						metadata: { legacy: true },
					};
					await create(matchPayload);
				}
				localStorage.setItem(LEGACY_LOADED_KEY, 'true');
				console.log('Legacy matches loaded into database');
			} catch (err) {
				console.warn('Failed to load legacy matches:', err);
			}
		};

		loadLegacyMatches();
	}, [create]);

	const handleWizardComplete = async (newMatch) => {
		setShowWizard(false);

		// Only auto-select if match is ready (has been processed)
		// Otherwise, show the match list where user can process it later
		if (newMatch.status === 'ready' && newMatch.params) {
			setShowList(false);
			setSelectedMatch(newMatch);
		} else {
			// Show match list for unprocessed matches
			setShowList(true);
		}
	};

	const handleSelectMatch = (match) => {
		setSelectedMatch(match);
		setShowList(false);
	};

	const handleCreateNew = () => {
		setShowList(false);
		setShowWizard(true);
	};

	const handleBrowseMatches = () => {
		setShowWizard(false);
		setShowList(true);
	};

	return (
		<div className="flex flex-col items-center w-full p-4 gap-4">
			<div className="text-center mb-2">
				<h1 className="text-4xl font-bold text-purple-600 mb-2">Video Stitcher</h1>
				<p className="text-muted-foreground">Create and manage your video stitching projects</p>
			</div>

			<div className="flex gap-2 mb-4">
				<Button onClick={handleCreateNew} disabled={showWizard}>
					+ Create New Match
				</Button>
				<Button variant="outline" onClick={handleBrowseMatches} disabled={showList}>
					Browse Matches
				</Button>
				<Button variant="outline" onClick={navigate.toProfiles}>
					Manage Lens Profiles
				</Button>
			</div>

			<Health />

			{showWizard ? (
				<MatchWizard onComplete={handleWizardComplete} onCancel={() => setShowWizard(false)} />
			) : showList ? (
				<MatchList onSelectMatch={handleSelectMatch} onCreateNew={handleCreateNew} />
			) : (
				<>
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
