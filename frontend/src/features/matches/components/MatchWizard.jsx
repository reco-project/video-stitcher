import React, { useState } from 'react';
import MatchCreationForm from './MatchCreationForm';
import { useMatchMutations } from '../hooks/useMatches';
import { processMatch } from '../api/matches';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function MatchWizard({ onComplete, onCancel }) {
	const [error, setError] = useState(null);
	const { create } = useMatchMutations();

	const handleFormSubmit = async (formData) => {
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
				name: formData.name,
				left_videos: formData.left_videos,
				right_videos: formData.right_videos,
				params: defaultParams,
				left_uniforms: buildUniforms(formData.leftProfile),
				right_uniforms: buildUniforms(formData.rightProfile),
				metadata: {
					left_profile_id: formData.leftProfile.id,
					right_profile_id: formData.rightProfile.id,
				},
				quality_settings: formData.qualitySettings,
			};

			const createdMatch = await create(matchPayload);
			
			// Start processing immediately after creation
			try {
				await processMatch(createdMatch.id);
			} catch (err) {
				console.error('Failed to start processing:', err);
				// Continue to processing page even if start fails - user can retry
			}
			
			onComplete(createdMatch);
		} catch (err) {
			setError(err.message || 'Failed to create match');
		}
	};

	const handleCancel = () => {
		onCancel();
	};

	return (
		<div className="w-full max-w-6xl space-y-6 relative">
			<MatchCreationForm onSubmit={handleFormSubmit} onCancel={handleCancel} />

			{/* Error Alert */}
			{error && (
				<Alert variant="destructive">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			)}
		</div>
	);
}
