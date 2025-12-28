import React, { useState, useEffect } from 'react';
import VideoImportStep from './VideoImportStep';
import ProfileAssignmentStep from './ProfileAssignmentStep';
import ProcessingStatus from './ProcessingStatus';
import { useMatchMutations } from '../hooks/useMatches';
import { useMatchProcessing } from '../hooks/useMatchProcessing';
import { getMatch } from '../api/matches';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

const DRAFT_KEY = 'matchWizardDraft';

export default function MatchWizard({ onComplete, onCancel }) {
	// Load draft from localStorage on mount
	const loadDraft = () => {
		try {
			const saved = localStorage.getItem(DRAFT_KEY);
			if (saved) {
				const draft = JSON.parse(saved);
				return {
					step: draft.step || 1,
					matchData: draft.matchData || {
						name: '',
						left_videos: [{ path: '', profile_id: null }],
						right_videos: [{ path: '', profile_id: null }],
					},
				};
			}
		} catch (err) {
			console.warn('Failed to load draft:', err);
		}
		return {
			step: 1,
			matchData: {
				name: '',
				left_videos: [{ path: '', profile_id: null }],
				right_videos: [{ path: '', profile_id: null }],
			},
		};
	};

	const initialState = loadDraft();
	const [step, setStep] = useState(initialState.step);
	const [matchData, setMatchData] = useState(initialState.matchData);
	const [error, setError] = useState(null);
	const [createdMatchId, setCreatedMatchId] = useState(null);
	const [showProcessing, setShowProcessing] = useState(false);

	const { create } = useMatchMutations();
	const processing = useMatchProcessing(createdMatchId, {
		pollInterval: 5000, // Poll every 5 seconds
		autoPoll: showProcessing && createdMatchId !== null,
	});

	// Save draft to localStorage whenever state changes
	useEffect(() => {
		try {
			localStorage.setItem(DRAFT_KEY, JSON.stringify({ step, matchData }));
		} catch (err) {
			console.warn('Failed to save draft:', err);
		}
	}, [step, matchData]);

	// Clear draft on unmount
	useEffect(() => {
		return () => {
			try {
				localStorage.removeItem(DRAFT_KEY);
			} catch (err) {
				console.warn('Failed to clear draft:', err);
			}
		};
	}, []);

	// Handle Escape key to cancel
	useEffect(() => {
		const handleKeyDown = (e) => {
			if (e.key === 'Escape') {
				if (confirm('Cancel match creation? Any unsaved progress will be lost.')) {
					onCancel();
				}
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [onCancel]);

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

			// Default calibration params - user can adjust later
			const defaultParams = {
				cameraAxisOffset: 0.23,
				intersect: 0.55,
				zRx: 0.0,
				xTy: 0.0,
				xRz: 0.0,
			};

			// Build uniforms from profile data
			const buildUniforms = (profile) => {
				// Validate profile structure
				if (
					!profile.resolution ||
					typeof profile.resolution.width !== 'number' ||
					typeof profile.resolution.height !== 'number'
				) {
					throw new Error(`Profile ${profile.id} has invalid resolution`);
				}

				if (
					!profile.camera_matrix ||
					typeof profile.camera_matrix.fx !== 'number' ||
					typeof profile.camera_matrix.fy !== 'number' ||
					typeof profile.camera_matrix.cx !== 'number' ||
					typeof profile.camera_matrix.cy !== 'number'
				) {
					throw new Error(`Profile ${profile.id} has invalid camera matrix`);
				}

				if (
					!profile.distortion_coeffs ||
					!Array.isArray(profile.distortion_coeffs) ||
					profile.distortion_coeffs.length !== 4 ||
					!profile.distortion_coeffs.every((c) => typeof c === 'number')
				) {
					throw new Error(`Profile ${profile.id} has invalid distortion coefficients`);
				}

				return {
					width: profile.resolution.width,
					height: profile.resolution.height,
					fx: profile.camera_matrix.fx,
					fy: profile.camera_matrix.fy,
					cx: profile.camera_matrix.cx,
					cy: profile.camera_matrix.cy,
					d: profile.distortion_coeffs,
				};
			};

			const matchPayload = {
				id,
				name: finalData.name,
				left_videos: finalData.left_videos,
				right_videos: finalData.right_videos,
				params: defaultParams,
				left_uniforms: buildUniforms(finalData.leftProfile),
				right_uniforms: buildUniforms(finalData.rightProfile),
				metadata: {
					left_profile_id: finalData.leftProfile.id,
					right_profile_id: finalData.rightProfile.id,
				},
			};

			const createdMatch = await create(matchPayload);
			// Clear draft on successful creation
			localStorage.removeItem(DRAFT_KEY);

			// Store match ID and show processing step
			setCreatedMatchId(createdMatch.id);
			setStep(3); // New processing step
			setShowProcessing(true);
		} catch (err) {
			setError(err.message || 'Failed to create match');
		}
	};

	const handleStartProcessing = async () => {
		try {
			await processing.startProcessing();
		} catch (err) {
			setError(err.message || 'Failed to start processing');
		}
	};

	const handleSkipProcessing = async () => {
		try {
			// Fetch the current match data to pass its status
			const match = await getMatch(createdMatchId);
			// Clear draft and complete without processing
			localStorage.removeItem(DRAFT_KEY);
			onComplete(match);
		} catch (err) {
			console.error('Failed to fetch match:', err);
			// Fallback: pass just the ID
			localStorage.removeItem(DRAFT_KEY);
			onComplete({ id: createdMatchId, status: 'pending' });
		}
	};

	const handleProcessingComplete = async () => {
		try {
			// Fetch the processed match data
			const match = await getMatch(createdMatchId);
			// Complete wizard with processed match
			localStorage.removeItem(DRAFT_KEY);
			onComplete(match);
		} catch (err) {
			console.error('Failed to fetch match:', err);
			// Fallback: pass just the ID with ready status
			localStorage.removeItem(DRAFT_KEY);
			onComplete({ id: createdMatchId, status: 'ready' });
		}
	};

	const handleBack = () => {
		setStep(1);
		setError(null);
	};

	return (
		<div className="w-full max-w-4xl">
			{/* Step Progress Indicator */}
			<div className="mb-6">
				<div className="flex items-center justify-center gap-2 mb-2">
					<div
						className={`flex items-center justify-center w-8 h-8 rounded-full ${step >= 1 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
					>
						1
					</div>
					<div className={`h-1 w-24 ${step >= 2 ? 'bg-primary' : 'bg-muted'}`} />
					<div
						className={`flex items-center justify-center w-8 h-8 rounded-full ${step >= 2 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
					>
						2
					</div>
					<div className={`h-1 w-24 ${step >= 3 ? 'bg-primary' : 'bg-muted'}`} />
					<div
						className={`flex items-center justify-center w-8 h-8 rounded-full ${step >= 3 ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
					>
						3
					</div>
				</div>
				<p className="text-center text-sm text-muted-foreground">
					Step {step} of 3: {step === 1 ? 'Import Videos' : step === 2 ? 'Assign Profiles' : 'Process Videos'}
				</p>
			</div>

			{error && (
				<Alert variant="destructive" className="mb-4">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			)}

			{step === 1 && <VideoImportStep onNext={handleStep1Complete} initialData={matchData} />}

			{step === 2 && (
				<ProfileAssignmentStep matchData={matchData} onNext={handleStep2Complete} onBack={handleBack} />
			)}

			{step === 3 && (
				<div className="space-y-4">
					<h2 className="text-2xl font-bold">Process Videos</h2>
					<p className="text-muted-foreground">
						Start video processing to synchronize, stack, and calibrate your cameras automatically.
					</p>

					{/* Processing Status */}
					{processing.status && <ProcessingStatus status={processing.status} />}

					{/* Action Buttons */}
					<div className="flex gap-2">
						{!processing.status || processing.status.status === 'pending' ? (
							<>
								<Button onClick={handleStartProcessing} disabled={processing.loading}>
									{processing.loading ? 'Starting...' : 'Start Processing'}
								</Button>
								<Button variant="outline" onClick={handleSkipProcessing}>
									Skip for Now
								</Button>
							</>
						) : processing.status.status === 'ready' ? (
							<Button onClick={handleProcessingComplete}>Continue to Match</Button>
						) : processing.status.status === 'error' ? (
							<>
								<Button onClick={handleStartProcessing}>Retry</Button>
								<Button variant="outline" onClick={handleSkipProcessing}>
									Skip for Now
								</Button>
							</>
						) : (
							<Button disabled>Processing...</Button>
						)}
						<Button variant="ghost" onClick={onCancel}>
							Cancel
						</Button>
					</div>
				</div>
			)}
		</div>
	);
}
