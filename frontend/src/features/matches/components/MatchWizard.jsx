import React, { useState } from 'react';
import MatchCreationForm from './MatchCreationForm';
import { useMatchMutations } from '../hooks/useMatches';
import { processMatch } from '../api/matches';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function MatchWizard({ onComplete, onCancel, initialMatch }) {
	const [error, setError] = useState(null);
	const { create, update } = useMatchMutations();
	const isEditMode = !!initialMatch;

	const handleFormSubmit = async (formData, startProcessing = true) => {
		try {
			setError(null);

			// Generate unique ID from timestamp
			const id = `match-${Date.now()}`;

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
			id: isEditMode ? initialMatch.id : id,
			name: formData.name,
			left_videos: formData.left_videos,
			right_videos: formData.right_videos,
			// params will be set after calibration completes
			left_uniforms: buildUniforms(formData.leftProfile),
			right_uniforms: buildUniforms(formData.rightProfile),
			metadata: {
				left_profile_id: formData.leftProfile.id,
				right_profile_id: formData.rightProfile.id,
			},
			quality_settings: formData.qualitySettings,
		// Only reset processing state when editing if we're going to reprocess
		...(isEditMode && startProcessing && {
			processing: {
				status: 'pending',
				step: null,
				message: null,
				error_message: null,
				error_code: null
			}
		})
	};

	const savedMatch = isEditMode 
		? await update(initialMatch.id, matchPayload)
		: await create(matchPayload);
		
		// Start processing if requested
		if (startProcessing) {
			try {
				await processMatch(savedMatch.id);
			} catch (err) {
				console.error('Failed to start processing:', err);
				// Continue even if start fails - user can retry
			}
		}
		
		onComplete(savedMatch, startProcessing);
	} catch (err) {
		setError(err.message || 'Failed to create match');
	}
};

const handleCancel = () => {
	onCancel();
};

return (
	<div className="w-full max-w-6xl space-y-6 relative pb-12">
		{/* Error Alert - Fixed at top for visibility */}
		{error && (
			<Alert variant="destructive" className="sticky top-4 z-50 shadow-lg">
				<AlertDescription className="font-medium">{error}</AlertDescription>
			</Alert>
		)}
		
		<MatchCreationForm 
			onSubmit={handleFormSubmit} 
			onCancel={handleCancel} 
			error={error}
			initialData={initialMatch}
		/>
	</div>
);
}
