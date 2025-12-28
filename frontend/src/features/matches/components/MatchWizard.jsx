import React, { useState, useEffect } from 'react';
import VideoImportStep from './VideoImportStep';
import ProfileAssignmentStep from './ProfileAssignmentStep';
import { useMatchMutations } from '../hooks/useMatches';
import { Alert, AlertDescription } from '@/components/ui/alert';

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

	const { create } = useMatchMutations();

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
			{/* Warning Banner */}
			<Alert variant="warning" className="mb-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-950">
				<AlertDescription className="text-yellow-800 dark:text-yellow-200">
					<strong>⚠️ Note:</strong> Backend video processing is not yet implemented. Matches will be created
					but videos will not be stitched automatically. The output video URL will need to be added manually.
				</AlertDescription>
			</Alert>

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
				</div>
				<p className="text-center text-sm text-muted-foreground">
					Step {step} of 2: {step === 1 ? 'Import Videos' : 'Assign Profiles'}
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
		</div>
	);
}
