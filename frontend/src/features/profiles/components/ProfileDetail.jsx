import React from 'react';
import { useProfile } from '../hooks/useProfiles';

export default function ProfileDetail({ profileId }) {
	const { profile, loading, error } = useProfile(profileId);

	if (!profileId) return <div className="text-gray-500">Select a profile to view details</div>;
	if (loading) return <div>Loading...</div>;
	if (error) return <div className="text-red-700">Error: {error}</div>;
	if (!profile) return null;

	return (
		<div className="w-full p-4 border rounded bg-gray-50">
			<h3 className="text-lg font-bold mb-4">Profile Details</h3>

			<div className="grid gap-3">
				<div>
					<label className="text-sm font-bold text-gray-600">ID</label>
					<div className="font-mono text-sm">{profile.id}</div>
				</div>

				<div>
					<label className="text-sm font-bold text-gray-600">Camera</label>
					<div>
						{profile.camera_brand} {profile.camera_model}
					</div>
				</div>

				{profile.lens_model && (
					<div>
						<label className="text-sm font-bold text-gray-600">Lens</label>
						<div>{profile.lens_model}</div>
					</div>
				)}

				<div>
					<label className="text-sm font-bold text-gray-600">Resolution</label>
					<div>
						{profile.resolution.width} × {profile.resolution.height}
					</div>
				</div>

				<div>
					<label className="text-sm font-bold text-gray-600">Distortion Model</label>
					<div className="font-mono text-sm">{profile.distortion_model}</div>
				</div>

				<div>
					<label className="text-sm font-bold text-gray-600">Camera Matrix</label>
					<div className="font-mono text-xs bg-white p-2 rounded">
						<div>fx: {profile.camera_matrix.fx.toFixed(2)}</div>
						<div>fy: {profile.camera_matrix.fy.toFixed(2)}</div>
						<div>cx: {profile.camera_matrix.cx.toFixed(2)}</div>
						<div>cy: {profile.camera_matrix.cy.toFixed(2)}</div>
					</div>
				</div>

				<div>
					<label className="text-sm font-bold text-gray-600">Distortion Coefficients</label>
					<div className="font-mono text-xs bg-white p-2 rounded">
						[{profile.distortion_coeffs.map((c) => c.toFixed(4)).join(', ')}]
					</div>
				</div>

				{profile.calib_dimension && (
					<div>
						<label className="text-sm font-bold text-gray-600">Calibration Dimension</label>
						<div className="text-sm">
							{profile.calib_dimension.width} × {profile.calib_dimension.height}
						</div>
					</div>
				)}

				{profile.note && (
					<div>
						<label className="text-sm font-bold text-gray-600">Note</label>
						<div className="text-sm text-gray-600">{profile.note}</div>
					</div>
				)}
			</div>
		</div>
	);
}
