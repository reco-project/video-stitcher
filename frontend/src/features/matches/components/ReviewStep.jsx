import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Pencil, Video, Camera, CheckCircle } from 'lucide-react';

export default function ReviewStep({ matchData, onConfirm, onBack, onEditVideos, onEditProfiles }) {
	const leftProfile = matchData.leftProfile;
	const rightProfile = matchData.rightProfile;

	const getFilename = (path) => {
		if (!path) return '';
		const parts = path.split(/[\\/]/);
		return parts[parts.length - 1];
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle className="flex items-center gap-2">
					<CheckCircle className="h-5 w-5 text-green-500" />
					Review & Confirm Match
				</CardTitle>
				<p className="text-sm text-muted-foreground mt-2">
					Review all match details before processing. Click edit to make changes.
				</p>
			</CardHeader>
			<CardContent className="space-y-6">
				{/* Match Name */}
				<div className="space-y-2">
					<div className="flex items-center justify-between">
						<h3 className="font-semibold text-lg">Match Name</h3>
						<Button type="button" variant="ghost" size="sm" onClick={onEditVideos}>
							<Pencil className="h-3 w-3 mr-1" />
							Edit
						</Button>
					</div>
					<div className="p-3 bg-muted rounded-lg">
						<p className="font-medium">{matchData.name || 'Untitled Match'}</p>
					</div>
				</div>

				{/* Videos Section */}
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					{/* Left Videos */}
					<div className="space-y-2">
						<div className="flex items-center justify-between">
							<h3 className="font-semibold flex items-center gap-2">
								<Video className="h-4 w-4" />
								Left Camera Videos
								<Badge variant="secondary">{matchData.left_videos?.length || 0}</Badge>
							</h3>
							<Button type="button" variant="ghost" size="sm" onClick={onEditVideos}>
								<Pencil className="h-3 w-3 mr-1" />
								Edit
							</Button>
						</div>
						<div className="p-3 bg-muted rounded-lg space-y-2 max-h-48 overflow-y-auto">
							{matchData.left_videos && matchData.left_videos.length > 0 ? (
								matchData.left_videos.map((video, idx) => (
									<div
										key={idx}
										className="text-sm py-1 px-2 bg-background rounded flex items-center gap-2"
									>
										<span className="text-muted-foreground font-mono">{idx + 1}.</span>
										<span className="truncate" title={video.path}>
											{getFilename(video.path)}
										</span>
									</div>
								))
							) : (
								<p className="text-sm text-muted-foreground">No videos selected</p>
							)}
						</div>
					</div>

					{/* Right Videos */}
					<div className="space-y-2">
						<div className="flex items-center justify-between">
							<h3 className="font-semibold flex items-center gap-2">
								<Video className="h-4 w-4" />
								Right Camera Videos
								<Badge variant="secondary">{matchData.right_videos?.length || 0}</Badge>
							</h3>
							<Button type="button" variant="ghost" size="sm" onClick={onEditVideos}>
								<Pencil className="h-3 w-3 mr-1" />
								Edit
							</Button>
						</div>
						<div className="p-3 bg-muted rounded-lg space-y-2 max-h-48 overflow-y-auto">
							{matchData.right_videos && matchData.right_videos.length > 0 ? (
								matchData.right_videos.map((video, idx) => (
									<div
										key={idx}
										className="text-sm py-1 px-2 bg-background rounded flex items-center gap-2"
									>
										<span className="text-muted-foreground font-mono">{idx + 1}.</span>
										<span className="truncate" title={video.path}>
											{getFilename(video.path)}
										</span>
									</div>
								))
							) : (
								<p className="text-sm text-muted-foreground">No videos selected</p>
							)}
						</div>
					</div>
				</div>

				{/* Profiles Section */}
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					{/* Left Profile */}
					<div className="space-y-2">
						<div className="flex items-center justify-between">
							<h3 className="font-semibold flex items-center gap-2">
								<Camera className="h-4 w-4" />
								Left Camera Profile
							</h3>
							<Button type="button" variant="ghost" size="sm" onClick={onEditProfiles}>
								<Pencil className="h-3 w-3 mr-1" />
								Edit
							</Button>
						</div>
						{leftProfile ? (
							<div className="p-3 bg-muted rounded-lg space-y-2">
								<div className="flex items-center justify-between">
									<span className="font-medium">
										{leftProfile.camera_brand} {leftProfile.camera_model}
									</span>
									{leftProfile.is_favorite && <span className="text-yellow-500">⭐</span>}
								</div>
								<div className="text-sm space-y-1 text-muted-foreground">
									<div>
										<span className="font-medium">Lens:</span>{' '}
										{leftProfile.lens_model || 'Standard'}
									</div>
									<div>
										<span className="font-medium">Resolution:</span> {leftProfile.resolution.width}x
										{leftProfile.resolution.height}
									</div>
									<div>
										<span className="font-medium">Model:</span>{' '}
										{leftProfile.distortion_model || 'fisheye_kb4'}
									</div>
								</div>
							</div>
						) : (
							<div className="p-3 bg-muted rounded-lg">
								<p className="text-sm text-muted-foreground">No profile selected</p>
							</div>
						)}
					</div>

					{/* Right Profile */}
					<div className="space-y-2">
						<div className="flex items-center justify-between">
							<h3 className="font-semibold flex items-center gap-2">
								<Camera className="h-4 w-4" />
								Right Camera Profile
							</h3>
							<Button type="button" variant="ghost" size="sm" onClick={onEditProfiles}>
								<Pencil className="h-3 w-3 mr-1" />
								Edit
							</Button>
						</div>
						{rightProfile ? (
							<div className="p-3 bg-muted rounded-lg space-y-2">
								<div className="flex items-center justify-between">
									<span className="font-medium">
										{rightProfile.camera_brand} {rightProfile.camera_model}
									</span>
									{rightProfile.is_favorite && <span className="text-yellow-500">⭐</span>}
								</div>
								<div className="text-sm space-y-1 text-muted-foreground">
									<div>
										<span className="font-medium">Lens:</span>{' '}
										{rightProfile.lens_model || 'Standard'}
									</div>
									<div>
										<span className="font-medium">Resolution:</span> {rightProfile.resolution.width}
										x{rightProfile.resolution.height}
									</div>
									<div>
										<span className="font-medium">Model:</span>{' '}
										{rightProfile.distortion_model || 'fisheye_kb4'}
									</div>
								</div>
							</div>
						) : (
							<div className="p-3 bg-muted rounded-lg">
								<p className="text-sm text-muted-foreground">No profile selected</p>
							</div>
						)}
					</div>
				</div>

				{/* Action Buttons */}
				<div className="flex justify-between pt-4 border-t">
					<Button type="button" variant="outline" onClick={onBack}>
						Back
					</Button>
					<Button onClick={onConfirm} size="lg" className="px-8">
						Create & Process Match
					</Button>
				</div>
			</CardContent>
		</Card>
	);
}
