import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import MatchListComponent from '@/features/matches/components/MatchList';
import { useMatchMutations } from '@/features/matches/hooks/useMatches';
import { processMatch } from '@/features/matches/api/matches';
import legacyMatches from '@/data/matches.js';

const LEGACY_LOADED_KEY = 'legacyMatchesLoaded';

export default function MatchList() {
	const navigate = useNavigate();
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
