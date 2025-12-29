import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { ArrowRight, Star } from 'lucide-react';
import { useBrands, useModels, useProfilesByBrandModel } from '@/features/profiles/hooks/useProfiles';
import { listFavoriteProfiles } from '@/features/profiles/api/profiles';
import { sortBrands, sortModels } from '@/lib/normalize';

export default function ProfileAssignmentStep({ matchData, onNext, onBack }) {
	const [leftBrand, setLeftBrand] = useState('');
	const [leftModel, setLeftModel] = useState('');
	const [leftProfileId, setLeftProfileId] = useState('');

	const [rightBrand, setRightBrand] = useState('');
	const [rightModel, setRightModel] = useState('');
	const [rightProfileId, setRightProfileId] = useState('');

	const [showLeftFavorites, setShowLeftFavorites] = useState(false);
	const [showRightFavorites, setShowRightFavorites] = useState(false);
	const [favoriteProfiles, setFavoriteProfiles] = useState([]);
	const [loadingFavorites, setLoadingFavorites] = useState(false);

	const [error, setError] = useState(null);

	// Hooks for left side
	const { brands: rawBrandsLeft } = useBrands();
	const { models: rawModelsLeft } = useModels(leftBrand);
	const { profiles: rawProfilesLeft } = useProfilesByBrandModel(leftBrand, leftModel);

	// Hooks for right side
	const { brands: rawBrandsRight } = useBrands();
	const { models: rawModelsRight } = useModels(rightBrand);
	const { profiles: rawProfilesRight } = useProfilesByBrandModel(rightBrand, rightModel);

	// Normalize and sort
	const brandsLeft = sortBrands(rawBrandsLeft);
	const modelsLeft = sortModels(rawModelsLeft);
	const profilesLeft = rawProfilesLeft.sort((a, b) => {
		if (a.is_favorite && !b.is_favorite) return -1;
		if (!a.is_favorite && b.is_favorite) return 1;
		return 0;
	});

	const brandsRight = sortBrands(rawBrandsRight);
	const modelsRight = sortModels(rawModelsRight);
	const profilesRight = rawProfilesRight.sort((a, b) => {
		if (a.is_favorite && !b.is_favorite) return -1;
		if (!a.is_favorite && b.is_favorite) return 1;
		return 0;
	});

	// Load favorites when toggled
	useEffect(() => {
		if (showLeftFavorites || showRightFavorites) {
			setLoadingFavorites(true);
			listFavoriteProfiles()
				.then(setFavoriteProfiles)
				.catch((err) => {
					console.error('Failed to load favorite profiles:', err);
					setError('Failed to load favorite profiles');
				})
				.finally(() => setLoadingFavorites(false));
		}
	}, [showLeftFavorites, showRightFavorites]);

	// Auto-select when there's only one option available in a select box
	useEffect(() => {
		if (!leftBrand && brandsLeft.length === 1) {
			setLeftBrand(brandsLeft[0]);
		}
	}, [brandsLeft, leftBrand]);

	useEffect(() => {
		if (leftBrand && !leftModel && modelsLeft.length === 1) {
			setLeftModel(modelsLeft[0]);
		}
	}, [modelsLeft, leftBrand, leftModel]);

	useEffect(() => {
		if (leftBrand && leftModel && !leftProfileId && profilesLeft.length === 1) {
			setLeftProfileId(profilesLeft[0].id);
		}
	}, [profilesLeft, leftBrand, leftModel, leftProfileId]);

	useEffect(() => {
		if (!rightBrand && brandsRight.length === 1) {
			setRightBrand(brandsRight[0]);
		}
	}, [brandsRight, rightBrand]);

	useEffect(() => {
		if (rightBrand && !rightModel && modelsRight.length === 1) {
			setRightModel(modelsRight[0]);
		}
	}, [modelsRight, rightBrand, rightModel]);

	useEffect(() => {
		if (rightBrand && rightModel && !rightProfileId && profilesRight.length === 1) {
			setRightProfileId(profilesRight[0].id);
		}
	}, [profilesRight, rightBrand, rightModel, rightProfileId]);

	const handleAutoAssign = () => {
		if (brandsLeft.length === 1) setLeftBrand(brandsLeft[0]);
		if (modelsLeft.length === 1) setLeftModel(modelsLeft[0]);
		if (profilesLeft.length === 1) setLeftProfileId(profilesLeft[0].id);

		if (brandsRight.length === 1) setRightBrand(brandsRight[0]);
		if (modelsRight.length === 1) setRightModel(modelsRight[0]);
		if (profilesRight.length === 1) setRightProfileId(profilesRight[0].id);
	};

	const handleCopyFromLeft = () => {
		if (!leftProfileId) {
			setError('Please select a left profile first');
			return;
		}
		// Copy the profile ID
		setRightProfileId(leftProfileId);
		// If in browse mode, also copy brand/model to maintain consistency
		if (!showLeftFavorites && leftBrand && leftModel) {
			setShowRightFavorites(false);
			setRightBrand(leftBrand);
			setRightModel(leftModel);
		}
		// If in favorites mode, switch right to favorites mode too
		if (showLeftFavorites) {
			setShowRightFavorites(true);
		}
		setError(null);
	};

	const handleNext = () => {
		if (!leftProfileId) {
			setError('Please select a lens profile for the left camera videos');
			return;
		}

		if (!rightProfileId) {
			setError('Please select a lens profile for the right camera videos');
			return;
		}

		// Find profiles from either browse mode or favorites mode
		const leftProfile = showLeftFavorites
			? favoriteProfiles.find((p) => p.id === leftProfileId)
			: profilesLeft.find((p) => p.id === leftProfileId);
		const rightProfile = showRightFavorites
			? favoriteProfiles.find((p) => p.id === rightProfileId)
			: profilesRight.find((p) => p.id === rightProfileId);

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
				<div className="flex items-center justify-between gap-4">
					<p className="text-sm text-muted-foreground mt-2">
						Select lens profiles for the left and right camera videos. The backend will use these to process
						and stitch the videos.
					</p>
					{/* Button to auto-assign profiles when only one option exists */}
					<Button size="sm" variant="outline" onClick={handleAutoAssign}>
						Auto Assign Profiles
					</Button>
				</div>
			</CardHeader>
			<CardContent className="space-y-6">
				{error && (
					<Alert variant="destructive">
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}

				{/* Left Camera Profile */}
				<div className="space-y-4 p-4 border rounded">
					<div className="flex items-center justify-between">
						<h3 className="font-semibold">
							Left Camera ({matchData.left_videos?.length || 0} video
							{matchData.left_videos?.length !== 1 ? 's' : ''})
						</h3>
						<Button
							type="button"
							size="sm"
							variant={showLeftFavorites ? 'default' : 'outline'}
							onClick={() => {
								setShowLeftFavorites(!showLeftFavorites);
								if (!showLeftFavorites) {
									// Reset brand/model when showing favorites
									setLeftBrand('');
									setLeftModel('');
								}
							}}
						>
							<Star className={`h-4 w-4 mr-1 ${showLeftFavorites ? 'fill-current' : ''}`} />
							Favorites
						</Button>
					</div>
					{matchData.left_videos && matchData.left_videos.length > 0 && (
						<div className="text-sm text-muted-foreground space-y-1 max-h-32 overflow-y-auto">
							{matchData.left_videos.map((video, idx) => (
								<div key={idx} className="truncate">
									{video.path}
								</div>
							))}
						</div>
					)}

					{showLeftFavorites ? (
						// Favorites mode
						<div>
							<Label htmlFor="left-profile-favorites">Profile</Label>
							<Select value={leftProfileId} onValueChange={setLeftProfileId}>
								<SelectTrigger id="left-profile-favorites" className="w-full">
									<SelectValue
										placeholder={
											loadingFavorites ? 'Loading favorites...' : 'Select favorite profile'
										}
									/>
								</SelectTrigger>
								<SelectContent>
									{favoriteProfiles.map((profile) => (
										<SelectItem key={profile.id} value={profile.id}>
											⭐ {profile.camera_brand} {profile.camera_model} -{' '}
											{profile.lens_model || 'Standard'} - {profile.resolution.width}x
											{profile.resolution.height}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					) : (
						// Browse mode
						<>
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
													{profile.is_favorite && '⭐ '}
													{profile.lens_model || 'Standard'} - {profile.resolution.width}x
													{profile.resolution.height}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								</div>
							)}
						</>
					)}

					{leftProfileId && (
						<>
							{(() => {
								const selectedProfile = showLeftFavorites
									? favoriteProfiles.find((p) => p.id === leftProfileId)
									: profilesLeft.find((p) => p.id === leftProfileId);
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
						</>
					)}
				</div>

				{/* Right Camera Profile */}
				<div className="space-y-4 p-4 border rounded">
					<div className="flex items-center justify-between">
						<h3 className="font-semibold">
							Right Camera ({matchData.right_videos?.length || 0} video
							{matchData.right_videos?.length !== 1 ? 's' : ''})
						</h3>
						<Button
							type="button"
							size="sm"
							variant={showRightFavorites ? 'default' : 'outline'}
							onClick={() => {
								setShowRightFavorites(!showRightFavorites);
								if (!showRightFavorites) {
									// Reset brand/model when showing favorites
									setRightBrand('');
									setRightModel('');
								}
							}}
						>
							<Star className={`h-4 w-4 mr-1 ${showRightFavorites ? 'fill-current' : ''}`} />
							Favorites
						</Button>
					</div>
					{matchData.right_videos && matchData.right_videos.length > 0 && (
						<div className="text-sm text-muted-foreground space-y-1 max-h-32 overflow-y-auto">
							{matchData.right_videos.map((video, idx) => (
								<div key={idx} className="truncate">
									{video.path}
								</div>
							))}
						</div>
					)}

					{showRightFavorites ? (
						// Favorites mode
						<div>
							<Label htmlFor="right-profile-favorites">Profile</Label>
							<Select value={rightProfileId} onValueChange={setRightProfileId}>
								<SelectTrigger id="right-profile-favorites" className="w-full">
									<SelectValue
										placeholder={
											loadingFavorites ? 'Loading favorites...' : 'Select favorite profile'
										}
									/>
								</SelectTrigger>
								<SelectContent>
									{favoriteProfiles.map((profile) => (
										<SelectItem key={profile.id} value={profile.id}>
											⭐ {profile.camera_brand} {profile.camera_model} -{' '}
											{profile.lens_model || 'Standard'} - {profile.resolution.width}x
											{profile.resolution.height}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					) : (
						// Browse mode
						<>
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
													{profile.is_favorite && '⭐ '}
													{profile.lens_model || 'Standard'} - {profile.resolution.width}x
													{profile.resolution.height}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								</div>
							)}
						</>
					)}

					<div className="mt-2 flex gap-2">
						<Button
							type="button"
							size="sm"
							variant="outline"
							onClick={handleCopyFromLeft}
							disabled={!leftProfileId}
							className="flex-1"
						>
							<ArrowRight className="h-4 w-4 mr-2" />
							Copy from Left Camera
						</Button>
					</div>

					{rightProfileId && (
						<>
							{(() => {
								const selectedProfile = showRightFavorites
									? favoriteProfiles.find((p) => p.id === rightProfileId)
									: profilesRight.find((p) => p.id === rightProfileId);
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
						</>
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
