import React, { useState } from 'react';
import { useProfile } from '../hooks/useProfiles';
import { toggleFavorite } from '../api/profiles';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { normalizeProfile } from '@/lib/normalize';
import { Star, Pencil, Trash2 } from 'lucide-react';
import { cn } from '@/lib/cn';

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
	const isOfficial = profile.metadata?.source === 'official';
	const isUser = profile.metadata?.source === 'user';
	const duplicatedFrom = profile.metadata?.duplicated_from;

	return (
		<Card className="w-full">
			<CardHeader className="pb-3">
				<div className="flex items-start justify-between gap-2">
					<div className="space-y-1.5 min-w-0 flex-1">
						<CardTitle className="text-base">Profile Details</CardTitle>
						<div className="flex gap-1.5 flex-wrap">
							{isOfficial && (
								<Badge variant="secondary" className="text-xs">
									📦 Official
								</Badge>
							)}
							{isUser && (
								<Badge variant="default" className="text-xs">
									👤 User
								</Badge>
							)}
							{duplicatedFrom && (
								<Badge variant="outline" className="text-xs" title={`Based on: ${duplicatedFrom}`}>
									📋 Copy
								</Badge>
							)}
						</div>
					</div>
					<div className="flex items-center gap-1 shrink-0 relative z-10">
						<Button
							size="icon"
							variant="ghost"
							className="h-8 w-8"
							onClick={handleToggleFavorite}
							disabled={favoriteLoading}
							title={profile.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
						>
							<Star
								className={cn(
									'h-4 w-4',
									profile.is_favorite ? 'fill-yellow-400 text-yellow-400' : 'text-muted-foreground'
								)}
							/>
						</Button>
						<Button
							size="icon"
							variant="ghost"
							className="h-8 w-8"
							onClick={() => onEdit && onEdit(profile)}
							title={isOfficial ? 'Edit will create a user copy' : 'Edit profile'}
						>
							<Pencil className="h-4 w-4" />
						</Button>
						{!isOfficial && (
							<Button
								size="icon"
								variant="ghost"
								className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
								onClick={() =>
									onDelete &&
									onDelete(profile.id, `${normalized.camera_brand} ${normalized.camera_model}`)
								}
								title="Delete profile"
							>
								<Trash2 className="h-4 w-4" />
							</Button>
						)}
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
