import React, { useState } from 'react';
import { useProfile } from '../hooks/useProfiles';
import { toggleFavorite } from '../api/profiles';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { normalizeProfile } from '@/lib/normalize';

export default function ProfileDetail({ profileId, onEdit, onDelete, onFavoriteToggle }) {
	const { profile, loading, error, refetch } = useProfile(profileId);
	const [favoriteLoading, setFavoriteLoading] = useState(false);

	const handleToggleFavorite = async () => {
		if (!profile) return;
		setFavoriteLoading(true);
		try {
			await toggleFavorite(profile.id, !profile.is_favorite);
			await refetch();
			if (onFavoriteToggle) onFavoriteToggle();
		} catch (err) {
			console.error('Failed to toggle favorite:', err);
		} finally {
			setFavoriteLoading(false);
		}
	};

	if (!profileId) return <div className="text-muted-foreground">Select a profile to view details</div>;
	if (loading) return <div>Loading...</div>;
	if (error) return <div className="text-red-700">Error: {error}</div>;
	if (!profile) return null;

	const normalized = normalizeProfile(profile);

	return (
		<Card className="w-full">
			<CardHeader>
				<div className="flex items-center justify-between">
					<div className="space-y-1">
						<CardTitle>Profile Details</CardTitle>
						<div className="flex gap-2">
							{profile.is_favorite && (
								<Badge variant="default" className="bg-yellow-500 hover:bg-yellow-600">
									⭐ Favorite
								</Badge>
							)}
							{profile.metadata?.is_custom && <Badge variant="default">Custom Profile</Badge>}
							{profile.metadata?.source === 'Gyroflow lens_profiles' && (
								<Badge variant="secondary">Gyroflow Official</Badge>
							)}
						</div>
					</div>
					<div className="flex gap-2">
						<Button
							size="sm"
							variant={profile.is_favorite ? 'outline' : 'default'}
							onClick={handleToggleFavorite}
							disabled={favoriteLoading}
						>
							{favoriteLoading ? '...' : profile.is_favorite ? '★ Unfavorite' : '☆ Favorite'}
						</Button>
						<Button size="sm" onClick={() => onEdit && onEdit(profile)}>
							Edit
						</Button>
						<Button size="sm" variant="destructive" onClick={() => onDelete && onDelete(profile.id)}>
							Delete
						</Button>
					</div>
				</div>
			</CardHeader>
			<CardContent className="space-y-4">
				<div>
					<label className="text-sm font-semibold text-muted-foreground">ID</label>
					<div className="font-mono text-sm">{profile.id}</div>
				</div>

				<Separator />

				<div>
					<label className="text-sm font-semibold text-muted-foreground">Camera</label>
					<div>
						{normalized.camera_brand} {normalized.camera_model}
					</div>
				</div>

				{profile.lens_model && (
					<>
						<Separator />
						<div>
							<label className="text-sm font-semibold text-muted-foreground">Lens</label>
							<div>{profile.lens_model}</div>
						</div>
					</>
				)}

				<Separator />

				<div>
					<label className="text-sm font-semibold text-muted-foreground">Resolution</label>
					<div>
						{profile.resolution.width} × {profile.resolution.height}
					</div>
				</div>

				<Separator />

				<div>
					<label className="text-sm font-semibold text-muted-foreground">Distortion Model</label>
					<div className="font-mono text-sm">{profile.distortion_model}</div>
				</div>

				<Separator />

				<div>
					<label className="text-sm font-semibold text-muted-foreground">Camera Matrix</label>
					<div className="font-mono text-xs bg-muted p-2 rounded space-y-1">
						<div>fx: {profile.camera_matrix.fx.toFixed(2)}</div>
						<div>fy: {profile.camera_matrix.fy.toFixed(2)}</div>
						<div>cx: {profile.camera_matrix.cx.toFixed(2)}</div>
						<div>cy: {profile.camera_matrix.cy.toFixed(2)}</div>
					</div>
				</div>

				<Separator />

				<div>
					<label className="text-sm font-semibold text-muted-foreground">Distortion Coefficients</label>
					<div className="font-mono text-xs bg-muted p-2 rounded">
						[{profile.distortion_coeffs.map((c) => c.toFixed(4)).join(', ')}]
					</div>
				</div>

				{profile.calib_dimension && (
					<>
						<Separator />
						<div>
							<label className="text-sm font-semibold text-muted-foreground">Calibration Dimension</label>
							<div className="text-sm">
								{profile.calib_dimension.width} × {profile.calib_dimension.height}
							</div>
						</div>
					</>
				)}

				{profile.note && (
					<>
						<Separator />
						<div>
							<label className="text-sm font-semibold text-muted-foreground">Note</label>
							<div className="text-sm">{profile.note}</div>
						</div>
					</>
				)}

				{profile.metadata && Object.keys(profile.metadata).length > 0 && (
					<>
						<Separator />
						<div>
							<label className="text-sm font-semibold text-muted-foreground">Metadata</label>
							<div className="text-xs bg-muted p-2 rounded space-y-1">
								{profile.metadata.created_at && (
									<div>
										<span className="font-semibold">Created:</span>{' '}
										{new Date(profile.metadata.created_at).toLocaleString()}
									</div>
								)}
								{profile.metadata.last_modified && (
									<div>
										<span className="font-semibold">Last Modified:</span>{' '}
										{new Date(profile.metadata.last_modified).toLocaleString()}
									</div>
								)}
								{profile.metadata.source && (
									<div>
										<span className="font-semibold">Source:</span> {profile.metadata.source}
									</div>
								)}
								{profile.metadata.calibrated_by && (
									<div>
										<span className="font-semibold">Calibrated By:</span>{' '}
										{profile.metadata.calibrated_by}
									</div>
								)}
								{profile.metadata.license && (
									<div>
										<span className="font-semibold">License:</span> {profile.metadata.license}
									</div>
								)}
							</div>
						</div>
					</>
				)}
			</CardContent>
		</Card>
	);
}
