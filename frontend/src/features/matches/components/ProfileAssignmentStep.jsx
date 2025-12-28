import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useBrands, useModels, useProfilesByBrandModel } from '@/features/profiles/hooks/useProfiles';

export default function ProfileAssignmentStep({ matchData, onNext, onBack }) {
	const [leftBrand, setLeftBrand] = useState('');
	const [leftModel, setLeftModel] = useState('');
	const [leftProfileId, setLeftProfileId] = useState('');

	const [rightBrand, setRightBrand] = useState('');
	const [rightModel, setRightModel] = useState('');
	const [rightProfileId, setRightProfileId] = useState('');

	const [error, setError] = useState(null);

	// Hooks for left side
	const { brands: brandsLeft } = useBrands();
	const { models: modelsLeft } = useModels(leftBrand);
	const { profiles: profilesLeft } = useProfilesByBrandModel(leftBrand, leftModel);

	// Hooks for right side
	const { brands: brandsRight } = useBrands();
	const { models: modelsRight } = useModels(rightBrand);
	const { profiles: profilesRight } = useProfilesByBrandModel(rightBrand, rightModel);

	const handleNext = () => {
		if (!leftProfileId) {
			setError('Please select a lens profile for the left camera videos');
			return;
		}

		if (!rightProfileId) {
			setError('Please select a lens profile for the right camera videos');
			return;
		}

		const leftProfile = profilesLeft.find((p) => p.id === leftProfileId);
		const rightProfile = profilesRight.find((p) => p.id === rightProfileId);

		if (!leftProfile || !rightProfile) {
			setError('Selected profiles not found');
			return;
		}

		onNext({
			...matchData,
			left_videos: matchData.left_videos.map((v) => ({
				...v,
				profile_id: leftProfileId,
			})),
			right_videos: matchData.right_videos.map((v) => ({
				...v,
				profile_id: rightProfileId,
			})),
			leftProfile,
			rightProfile,
		});
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle>Step 2: Assign Lens Profiles</CardTitle>
				<p className="text-sm text-muted-foreground mt-2">
					Select lens profiles for the left and right camera videos. The backend will use these to process and
					stitch the videos.
				</p>
			</CardHeader>
			<CardContent className="space-y-6">
				{error && (
					<Alert variant="destructive">
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}

				{/* Left Camera Profile */}
				<div className="space-y-4 p-4 border rounded">
					<h3 className="font-semibold">
						Left Camera ({matchData.left_videos?.length || 0} video
						{matchData.left_videos?.length !== 1 ? 's' : ''})
					</h3>
					{matchData.left_videos && matchData.left_videos.length > 0 && (
						<div className="text-sm text-muted-foreground space-y-1 max-h-32 overflow-y-auto">
							{matchData.left_videos.map((video, idx) => (
								<div key={idx} className="truncate">
									{video.path}
								</div>
							))}
						</div>
					)}

					<div>
						<Label htmlFor="left-brand">Brand</Label>
						<Select
							value={leftBrand}
							onValueChange={(value) => {
								setLeftBrand(value);
								setLeftModel('');
								setLeftProfileId('');
							}}
						>
							<SelectTrigger id="left-brand" className="w-full">
								<SelectValue placeholder="Select brand" />
							</SelectTrigger>
							<SelectContent>
								{brandsLeft.map((brand) => (
									<SelectItem key={brand} value={brand}>
										{brand}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</div>

					{leftBrand && (
						<div>
							<Label htmlFor="left-model">Model</Label>
							<Select
								value={leftModel}
								onValueChange={(value) => {
									setLeftModel(value);
									setLeftProfileId('');
								}}
							>
								<SelectTrigger id="left-model" className="w-full">
									<SelectValue placeholder="Select model" />
								</SelectTrigger>
								<SelectContent>
									{modelsLeft.map((model) => (
										<SelectItem key={model} value={model}>
											{model}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					)}

					{leftBrand && leftModel && (
						<div>
							<Label htmlFor="left-profile">Profile</Label>
							<Select value={leftProfileId} onValueChange={setLeftProfileId}>
								<SelectTrigger id="left-profile" className="w-full">
									<SelectValue placeholder="Select profile" />
								</SelectTrigger>
								<SelectContent>
									{profilesLeft.map((profile) => (
										<SelectItem key={profile.id} value={profile.id}>
											{profile.lens_model || 'Standard'} - {profile.resolution.width}x
											{profile.resolution.height}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
							{leftProfileId &&
								(() => {
									const selectedProfile = profilesLeft.find((p) => p.id === leftProfileId);
									return selectedProfile ? (
										<div className="mt-2 p-3 bg-muted rounded text-xs space-y-1">
											<div className="font-semibold">Profile Details:</div>
											<div>
												Resolution: {selectedProfile.resolution.width}x
												{selectedProfile.resolution.height}
											</div>
											<div>Lens: {selectedProfile.lens_model || 'Standard'}</div>
											{selectedProfile.distortion_coefficients && (
												<div>
													Distortion: [
													{selectedProfile.distortion_coefficients
														.slice(0, 3)
														.map((d) => d.toFixed(3))
														.join(', ')}
													...]
												</div>
											)}
										</div>
									) : null;
								})()}
						</div>
					)}
				</div>

				{/* Right Camera Profile */}
				<div className="space-y-4 p-4 border rounded">
					<h3 className="font-semibold">
						Right Camera ({matchData.right_videos?.length || 0} video
						{matchData.right_videos?.length !== 1 ? 's' : ''})
					</h3>
					{matchData.right_videos && matchData.right_videos.length > 0 && (
						<div className="text-sm text-muted-foreground space-y-1 max-h-32 overflow-y-auto">
							{matchData.right_videos.map((video, idx) => (
								<div key={idx} className="truncate">
									{video.path}
								</div>
							))}
						</div>
					)}

					<div>
						<Label htmlFor="right-brand">Brand</Label>
						<Select
							value={rightBrand}
							onValueChange={(value) => {
								setRightBrand(value);
								setRightModel('');
								setRightProfileId('');
							}}
						>
							<SelectTrigger id="right-brand" className="w-full">
								<SelectValue placeholder="Select brand" />
							</SelectTrigger>
							<SelectContent>
								{brandsRight.map((brand) => (
									<SelectItem key={brand} value={brand}>
										{brand}
									</SelectItem>
								))}
							</SelectContent>
						</Select>
					</div>

					{rightBrand && (
						<div>
							<Label htmlFor="right-model">Model</Label>
							<Select
								value={rightModel}
								onValueChange={(value) => {
									setRightModel(value);
									setRightProfileId('');
								}}
							>
								<SelectTrigger id="right-model" className="w-full">
									<SelectValue placeholder="Select model" />
								</SelectTrigger>
								<SelectContent>
									{modelsRight.map((model) => (
										<SelectItem key={model} value={model}>
											{model}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					)}

					{rightBrand && rightModel && (
						<div>
							<Label htmlFor="right-profile">Profile</Label>
							<Select value={rightProfileId} onValueChange={setRightProfileId}>
								<SelectTrigger id="right-profile" className="w-full">
									<SelectValue placeholder="Select profile" />
								</SelectTrigger>
								<SelectContent>
									{profilesRight.map((profile) => (
										<SelectItem key={profile.id} value={profile.id}>
											{profile.lens_model || 'Standard'} - {profile.resolution.width}x
											{profile.resolution.height}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
							{rightProfileId &&
								(() => {
									const selectedProfile = profilesRight.find((p) => p.id === rightProfileId);
									return selectedProfile ? (
										<div className="mt-2 p-3 bg-muted rounded text-xs space-y-1">
											<div className="font-semibold">Profile Details:</div>
											<div>
												Resolution: {selectedProfile.resolution.width}x
												{selectedProfile.resolution.height}
											</div>
											<div>Lens: {selectedProfile.lens_model || 'Standard'}</div>
											{selectedProfile.distortion_coefficients && (
												<div>
													Distortion: [
													{selectedProfile.distortion_coefficients
														.slice(0, 3)
														.map((d) => d.toFixed(3))
														.join(', ')}
													...]
												</div>
											)}
										</div>
									) : null;
								})()}
						</div>
					)}
				</div>

				<div className="flex justify-between">
					<Button type="button" variant="outline" onClick={onBack}>
						Back
					</Button>
					<Button onClick={handleNext}>Create Match</Button>
				</div>
			</CardContent>
		</Card>
	);
}
