import React from 'react';
import { useProfiles } from '../hooks/useProfiles';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

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
				<div className="grid gap-2">
					{profiles.map((profile) => (
						<Card
							key={profile.id}
							className="cursor-pointer hover:bg-accent transition-colors"
							onClick={() => onSelect && onSelect(profile)}
						>
							<CardHeader className="p-3">
								<CardTitle className="text-base">
									{profile.camera_brand} {profile.camera_model}
								</CardTitle>
							</CardHeader>
							<CardContent className="p-3 pt-0">
								{profile.lens_model && (
									<div className="text-sm text-muted-foreground">{profile.lens_model}</div>
								)}
								<div className="text-xs text-muted-foreground">
									{profile.resolution.width}x{profile.resolution.height}
								</div>
							</CardContent>
						</Card>
					))}
				</div>
			)}
		</div>
	);
}
