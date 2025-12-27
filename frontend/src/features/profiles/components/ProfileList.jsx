import React from 'react';
import { useProfiles } from '../hooks/useProfiles';

export default function ProfileList({ onSelect }) {
	const { profiles, loading, error } = useProfiles();

	if (loading) return <div>Loading profiles...</div>;
	if (error) return <div className="text-red-700">Error: {error}</div>;

	return (
		<div className="w-full">
			<h3 className="text-lg font-bold mb-2">Lens Profiles</h3>
			{profiles.length === 0 ? (
				<p className="text-gray-500">No profiles available</p>
			) : (
				<div className="grid gap-2">
					{profiles.map((profile) => (
						<div
							key={profile.id}
							className="p-3 border rounded hover:bg-gray-50 cursor-pointer transition"
							onClick={() => onSelect && onSelect(profile)}
						>
							<div className="font-bold">
								{profile.camera_brand} {profile.camera_model}
							</div>
							{profile.lens_model && <div className="text-sm text-gray-600">{profile.lens_model}</div>}
							<div className="text-xs text-gray-500">
								{profile.resolution.width}x{profile.resolution.height}
							</div>
						</div>
					))}
				</div>
			)}
		</div>
	);
}
