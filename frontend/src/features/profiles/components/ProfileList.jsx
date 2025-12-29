import React from 'react';
import { useProfiles } from '../hooks/useProfiles';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { normalizeProfile } from '@/lib/normalize';

export default function ProfileList({ onSelect }) {
	const { profiles, loading, error } = useProfiles();

	if (loading) return <div>Loading profiles...</div>;
	if (error) return <div className="text-red-700">Error: {error}</div>;

	return (
		<div className="w-full">
			<h3 className="text-lg font-bold mb-2">Lens Profiles</h3>
			{profiles.length === 0 ? (
				<p className="text-muted-foreground">No profiles available</p>
			) : (
				<div className="grid gap-1">
					{profiles.map((profile) => {
						const normalized = normalizeProfile(profile);
						return (
							<Card
								key={profile.id}
								className="cursor-pointer hover:bg-accent transition-colors py-0 gap-0"
								onClick={() => onSelect && onSelect(profile)}
							>
								<CardContent className="px-2.5 py-1">
									<div className="flex items-center justify-between gap-2">
										<div className="text-sm font-semibold flex items-center gap-1.5">
											{profile.is_favorite && <span className="text-yellow-500 text-xs">⭐</span>}
											{normalized.camera_brand} {normalized.camera_model}
										</div>
										<div className="text-xs text-muted-foreground">
											{profile.resolution.width}×{profile.resolution.height}
										</div>
									</div>
									{profile.lens_model && (
										<div className="text-xs text-muted-foreground mt-0.5">{profile.lens_model}</div>
									)}
								</CardContent>
							</Card>
						);
					})}
				</div>
			)}
		</div>
	);
}
