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
			setError('Please select a lens profile for the left videos');
			return;
		}

		if (!rightProfileId) {
			setError('Please select a lens profile for the right videos');
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
		});
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle>Step 2: Assign Lens Profiles</CardTitle>
			</CardHeader>
			<CardContent className="space-y-6">
				{error && (
					<Alert variant="destructive">
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}

				{/* Left Videos Profile */}
				<div className="space-y-4 p-4 border rounded">
					<h3 className="font-semibold">
						Left Camera ({matchData.left_videos.length} video{matchData.left_videos.length > 1 ? 's' : ''})
					</h3>
					<div className="text-sm text-muted-foreground space-y-1">
						{matchData.left_videos.map((video, idx) => (
							<div key={idx}>{video.path}</div>
						))}
					</div>

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
						</div>
					)}
				</div>

				{/* Right Videos Profile */}
				<div className="space-y-4 p-4 border rounded">
					<h3 className="font-semibold">
						Right Camera ({matchData.right_videos.length} video
						{matchData.right_videos.length > 1 ? 's' : ''})
					</h3>
					<div className="text-sm text-muted-foreground space-y-1">
						{matchData.right_videos.map((video, idx) => (
							<div key={idx}>{video.path}</div>
						))}
					</div>

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
