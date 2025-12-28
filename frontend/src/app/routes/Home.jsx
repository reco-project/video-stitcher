import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import Viewer from '@/features/viewer/components/Viewer.jsx';
import MatchWizard from '@/features/matches/components/MatchWizard';
import MatchList from '@/features/matches/components/MatchList';
import { useMatchMutations } from '@/features/matches/hooks/useMatches';
import legacyMatches from '@/data/matches.js';

const LEGACY_LOADED_KEY = 'legacyMatchesLoaded';

export default function Home() {
	const [searchParams, setSearchParams] = useSearchParams();
	const [selectedMatch, setSelectedMatch] = useState(null);
	const { create } = useMatchMutations();

	// Determine current view from URL params
	const mode = searchParams.get('view') || 'list';
	const showWizard = mode === 'create';
	const showList = mode === 'list';
	const showViewer = mode === 'viewer';

	// Helper to update view
	const setView = (newView) => {
		setSearchParams({ view: newView });
	};

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
		// Auto-navigate to viewer if match is ready, otherwise show list
		if (newMatch.status === 'ready' && newMatch.params) {
			setSelectedMatch(newMatch);
			setView('viewer');
		} else {
			// Show match list for incomplete matches
			setView('list');
		}
	};

	const handleSelectMatch = (match) => {
		setSelectedMatch(match);
		setView('viewer');
	};

	const handleCreateNew = () => {
		setView('create');
	};

	return (
		<div className="w-full h-full flex flex-col items-center justify-start px-6 py-6">
			<div className="w-full max-w-6xl">
				{/* Page Content */}
				{showWizard ? (
					<div className="mt-6 flex justify-center">
						<MatchWizard onComplete={handleWizardComplete} onCancel={() => setView('list')} />
					</div>
				) : showList ? (
					<div className="mt-6 flex justify-center">
						<MatchList onSelectMatch={handleSelectMatch} onCreateNew={handleCreateNew} />
					</div>
				) : showViewer ? (
					<div className="mt-6 w-full flex justify-center">
						{selectedMatch && (
							<section className="w-full max-w-6xl aspect-video flex flex-col gap-4">
								<button
									onClick={() => setView('list')}
									className="text-sm text-muted-foreground hover:text-foreground transition-colors"
								>
									← Back to Matches
								</button>
								<Viewer key={selectedMatch.id} selectedMatch={selectedMatch} />
							</section>
						)}
					</div>
				) : null}
			</div>
		</div>
	);
}
