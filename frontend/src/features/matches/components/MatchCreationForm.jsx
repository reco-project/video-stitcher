import React, { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Video, Camera, Star, ArrowRight, Trash2, GripVertical } from 'lucide-react';
import { useBrands, useModels, useProfilesByBrandModel } from '@/features/profiles/hooks/useProfiles';
import { listFavoriteProfiles } from '@/features/profiles/api/profiles';
import { sortBrands, sortModels } from '@/lib/normalize';

const DRAFT_KEY = 'matchCreationDraft';

const loadDraft = () => {
	try {
		const draft = localStorage.getItem(DRAFT_KEY);
		return draft ? JSON.parse(draft) : null;
	} catch (err) {
		console.warn('Failed to load draft:', err);
		return null;
	}
};

const saveDraft = (data) => {
	try {
		localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
	} catch (err) {
		console.warn('Failed to save draft:', err);
	}
};

export default function MatchCreationForm({ onSubmit, onCancel, initialData }) {
	const draft = loadDraft();
	const [name, setName] = useState(initialData?.name || draft?.name || '');

	// Video paths
	const [leftVideoPaths, setLeftVideoPaths] = useState(
		initialData?.left_videos?.map((v) => v.path) || draft?.leftVideoPaths || ['']
	);
	const [rightVideoPaths, setRightVideoPaths] = useState(
		initialData?.right_videos?.map((v) => v.path) || draft?.rightVideoPaths || ['']
	);
	const [leftMetadata, setLeftMetadata] = useState([]);
	const [rightMetadata, setRightMetadata] = useState([]);
	const [draggedIndex, setDraggedIndex] = useState(null);
	const [draggedSide, setDraggedSide] = useState(null);

	// Profile selection - Left
	const [showLeftFavorites, setShowLeftFavorites] = useState(draft?.showLeftFavorites || false);
	const [leftBrand, setLeftBrand] = useState(draft?.leftBrand || '');
	const [leftModel, setLeftModel] = useState(draft?.leftModel || '');
	const [leftProfileId, setLeftProfileId] = useState(draft?.leftProfileId || '');

	// Profile selection - Right
	const [showRightFavorites, setShowRightFavorites] = useState(draft?.showRightFavorites || false);
	const [rightBrand, setRightBrand] = useState(draft?.rightBrand || '');
	const [rightModel, setRightModel] = useState(draft?.rightModel || '');
	const [rightProfileId, setRightProfileId] = useState(draft?.rightProfileId || '');

	// Favorites
	const [favoriteProfiles, setFavoriteProfiles] = useState([]);
	const [loadingFavorites, setLoadingFavorites] = useState(false);

	const [error, setError] = useState(null);
	const [isSubmitting, setIsSubmitting] = useState(false);

	// Hooks for left side
	const { brands: rawBrandsLeft } = useBrands();
	const { models: rawModelsLeft } = useModels(leftBrand);
	const { profiles: rawProfilesLeft } = useProfilesByBrandModel(leftBrand, leftModel);

	// Hooks for right side
	const { brands: rawBrandsRight } = useBrands();
	const { models: rawModelsRight } = useModels(rightBrand);
	const { profiles: rawProfilesRight } = useProfilesByBrandModel(rightBrand, rightModel);

	// Sort and normalize
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

	// Save draft to localStorage with debounce to prevent focus interruption
	useEffect(() => {
		const timeoutId = setTimeout(() => {
			const draftData = {
				name,
				leftVideoPaths,
				rightVideoPaths,
				showLeftFavorites,
				leftBrand,
				leftModel,
				leftProfileId,
				showRightFavorites,
				rightBrand,
				rightModel,
				rightProfileId,
			};
			saveDraft(draftData);
		}, 500); // Debounce by 500ms

		return () => clearTimeout(timeoutId);
	}, [
		name,
		leftVideoPaths,
		rightVideoPaths,
		showLeftFavorites,
		leftBrand,
		leftModel,
		leftProfileId,
		showRightFavorites,
		rightBrand,
		rightModel,
		rightProfileId,
	]);

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

	const loadMetadata = useCallback(async (filePath, side, index) => {
		if (!filePath || !window.electronAPI?.getFileMetadata) return;

		try {
			const metadata = await window.electronAPI.getFileMetadata(filePath);
			if (metadata) {
				if (side === 'left') {
					setLeftMetadata((prev) => {
						const newMeta = [...prev];
						newMeta[index] = metadata;
						return newMeta;
					});
				} else {
					setRightMetadata((prev) => {
						const newMeta = [...prev];
						newMeta[index] = metadata;
						return newMeta;
					});
				}
			}
		} catch (err) {
			console.warn('Failed to load metadata:', err);
		}
	}, []);

	// Reload metadata for persisted video paths on mount
	useEffect(() => {
		leftVideoPaths.forEach((path, index) => {
			if (path && path.trim()) {
				loadMetadata(path, 'left', index);
			}
		});
		rightVideoPaths.forEach((path, index) => {
			if (path && path.trim()) {
				loadMetadata(path, 'right', index);
			}
		});
	}, []);

	const handleSelectFile = async (side, index) => {
		try {
			if (!window.electronAPI || !window.electronAPI.selectVideoFile) {
				throw new Error('File selection not available. Please run in Electron.');
			}

			const filePath = await window.electronAPI.selectVideoFile();

			if (filePath) {
				if (side === 'left') {
					const newPaths = [...leftVideoPaths];
					newPaths[index] = filePath;
					// Auto-add empty field if last one was filled
					if (index === leftVideoPaths.length - 1) {
						newPaths.push('');
					}
					setLeftVideoPaths(newPaths);
					loadMetadata(filePath, side, index);
				} else {
					const newPaths = [...rightVideoPaths];
					newPaths[index] = filePath;
					// Auto-add empty field if last one was filled
					if (index === rightVideoPaths.length - 1) {
						newPaths.push('');
					}
					setRightVideoPaths(newPaths);
					loadMetadata(filePath, side, index);
				}
				setError(null);
			}
		} catch (err) {
			setError(err.message);
		}
	};

	const handleRemoveVideo = (side, index) => {
		if (side === 'left') {
			setLeftVideoPaths(leftVideoPaths.filter((_, i) => i !== index));
			setLeftMetadata(leftMetadata.filter((_, i) => i !== index));
			// Ensure at least one empty field
			if (leftVideoPaths.length === 1) {
				setLeftVideoPaths(['']);
			}
		} else if (side === 'right') {
			setRightVideoPaths(rightVideoPaths.filter((_, i) => i !== index));
			setRightMetadata(rightMetadata.filter((_, i) => i !== index));
			// Ensure at least one empty field
			if (rightVideoPaths.length === 1) {
				setRightVideoPaths(['']);
			}
		}
	};

	const handleDragStart = (side, index) => {
		setDraggedIndex(index);
		setDraggedSide(side);
	};

	const handleDragOver = (e) => {
		e.preventDefault();
	};

	const handleDrop = (side, dropIndex) => {
		if (draggedSide !== side || draggedIndex === null || draggedIndex === dropIndex) {
			setDraggedIndex(null);
			setDraggedSide(null);
			return;
		}

		if (side === 'left') {
			const newPaths = [...leftVideoPaths];
			const newMetadata = [...leftMetadata];
			const [movedPath] = newPaths.splice(draggedIndex, 1);
			const [movedMeta] = newMetadata.splice(draggedIndex, 1);
			newPaths.splice(dropIndex, 0, movedPath);
			newMetadata.splice(dropIndex, 0, movedMeta);
			setLeftVideoPaths(newPaths);
			setLeftMetadata(newMetadata);
		} else {
			const newPaths = [...rightVideoPaths];
			const newMetadata = [...rightMetadata];
			const [movedPath] = newPaths.splice(draggedIndex, 1);
			const [movedMeta] = newMetadata.splice(draggedIndex, 1);
			newPaths.splice(dropIndex, 0, movedPath);
			newMetadata.splice(dropIndex, 0, movedMeta);
			setRightVideoPaths(newPaths);
			setRightMetadata(newMetadata);
		}

		setDraggedIndex(null);
		setDraggedSide(null);
	};

	const calculateTotals = (metadata) => {
		const totalSize = metadata.reduce((sum, m) => sum + (m?.size || 0), 0);
		const totalDuration = metadata.reduce((sum, m) => sum + (m?.duration || 0), 0);
		return {
			size: totalSize,
			sizeFormatted: formatBytes(totalSize),
			duration: totalDuration,
			durationFormatted: formatDuration(totalDuration),
		};
	};

	const formatBytes = (bytes) => {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
	};

	const formatDuration = (seconds) => {
		if (!seconds) return '0s';
		const hours = Math.floor(seconds / 3600);
		const minutes = Math.floor((seconds % 3600) / 60);
		const secs = Math.floor(seconds % 60);
		if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
		if (minutes > 0) return `${minutes}m ${secs}s`;
		return `${secs}s`;
	};

	const handleCopyFromLeft = () => {
		if (!leftProfileId) {
			setError('Please select a left profile first');
			return;
		}
		setRightProfileId(leftProfileId);
		if (showLeftFavorites) {
			setShowRightFavorites(true);
		} else if (leftBrand && leftModel) {
			setShowRightFavorites(false);
			setRightBrand(leftBrand);
			setRightModel(leftModel);
		}
		setError(null);
	};

	const handleSubmit = async () => {
		// Validation
		if (!name.trim()) {
			setError('Please enter a match name');
			return;
		}

		const validLeftPaths = leftVideoPaths.filter((p) => p.trim());
		const validRightPaths = rightVideoPaths.filter((p) => p.trim());

		if (validLeftPaths.length === 0) {
			setError('Please select at least one left camera video');
			return;
		}

		if (validRightPaths.length === 0) {
			setError('Please select at least one right camera video');
			return;
		}

		if (!leftProfileId) {
			setError('Please select a lens profile for the left camera');
			return;
		}

		if (!rightProfileId) {
			setError('Please select a lens profile for the right camera');
			return;
		}

		// Find profiles
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

		setIsSubmitting(true);
		setError(null);

		try {
			await onSubmit({
				name: name.trim(),
				left_videos: validLeftPaths.map((path) => ({ path, profile_id: leftProfileId })),
				right_videos: validRightPaths.map((path) => ({ path, profile_id: rightProfileId })),
				leftProfile,
				rightProfile,
			});
			// Clear draft on successful submission
			try {
				localStorage.removeItem(DRAFT_KEY);
			} catch (err) {
				console.warn('Failed to clear draft:', err);
			}
		} catch (err) {
			setError(err.message || 'Failed to create match');
			setIsSubmitting(false);
		}
	};

	const handleCancel = () => {
		// Clear draft when canceling
		try {
			localStorage.removeItem(DRAFT_KEY);
		} catch (err) {
			console.warn('Failed to clear draft:', err);
		}
		onCancel();
	};

	return (
		<div className="w-full max-w-6xl space-y-6">
			{/* Header */}
			<div>
				<h1 className="text-3xl font-bold">Create New Match</h1>
				<p className="text-muted-foreground mt-2">
					Configure your match by selecting videos and assigning lens profiles for both cameras.
				</p>
			</div>

			{error && (
				<Alert variant="destructive">
					<AlertDescription>{error}</AlertDescription>
				</Alert>
			)}

			{/* Match Name */}
			<Card>
				<CardHeader>
					<CardTitle>Match Name</CardTitle>
				</CardHeader>
				<CardContent>
					<Input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder="e.g., Concert 2025-12-29"
						className="text-lg"
					/>
				</CardContent>
			</Card>

			{/* Videos and Profiles Grid */}
			<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
				{/* Left Camera */}
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center gap-2">
							<Video className="h-5 w-5" />
							Left Camera
						</CardTitle>
					</CardHeader>
					<CardContent className="space-y-4">
						{/* Videos */}
						<div className="space-y-3">
							<div className="flex items-center justify-between">
								<Label className="text-base font-semibold">Videos</Label>
								{leftMetadata.filter(Boolean).length > 0 && (
									<div className="text-xs text-muted-foreground">
										{leftMetadata.filter(Boolean).length} video(s) •{' '}
										{calculateTotals(leftMetadata).durationFormatted} •{' '}
										{calculateTotals(leftMetadata).sizeFormatted}
									</div>
								)}
							</div>
							{leftVideoPaths.map((path, index) => (
								<div
									key={index}
									className="space-y-1"
									draggable={!!path}
									onDragStart={() => handleDragStart('left', index)}
									onDragOver={handleDragOver}
									onDrop={() => handleDrop('left', index)}
								>
									<div className="flex gap-2 items-center">
										<GripVertical
											className={`h-4 w-4 ${path ? 'text-muted-foreground cursor-grab' : 'text-muted-foreground/30'}`}
										/>
										<Input
											type="text"
											value={path}
											onChange={(e) => {
												const newPaths = [...leftVideoPaths];
												newPaths[index] = e.target.value;
												setLeftVideoPaths(newPaths);
											}}
											placeholder={
												index === leftVideoPaths.length - 1 ? 'Add video...' : 'Video file path'
											}
											className="flex-1"
										/>
										<Button type="button" size="sm" onClick={() => handleSelectFile('left', index)}>
											Browse
										</Button>
										{path && (
											<Button
												type="button"
												variant="ghost"
												size="sm"
												onClick={() => handleRemoveVideo('left', index)}
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										)}
									</div>
									{leftMetadata[index] && (
										<div className="text-xs text-muted-foreground pl-6 space-x-2">
											<span>{leftMetadata[index].name}</span>
											{leftMetadata[index].duration && (
												<span>• {formatDuration(leftMetadata[index].duration)}</span>
											)}
											<span>• {leftMetadata[index].sizeFormatted}</span>
										</div>
									)}
								</div>
							))}
						</div>

						{/* Profile Selection */}
						<div className="space-y-3 pt-4 border-t">
							<div className="flex items-center justify-between">
								<Label className="text-base font-semibold">
									<Camera className="h-4 w-4 inline mr-1" />
									Lens Profile
								</Label>
								<Button
									type="button"
									size="sm"
									variant={showLeftFavorites ? 'default' : 'outline'}
									onClick={() => {
										setShowLeftFavorites(!showLeftFavorites);
										if (!showLeftFavorites) {
											setLeftBrand('');
											setLeftModel('');
										}
									}}
								>
									<Star className={`h-3 w-3 mr-1 ${showLeftFavorites ? 'fill-current' : ''}`} />
									Favorites
								</Button>
							</div>

							{showLeftFavorites ? (
								<div>
									<Label>Profile</Label>
									<Select value={leftProfileId} onValueChange={setLeftProfileId}>
										<SelectTrigger>
											<SelectValue
												placeholder={
													loadingFavorites ? 'Loading...' : 'Select favorite profile'
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
								<>
									<div>
										<Label>Brand</Label>
										<Select
											value={leftBrand}
											onValueChange={(value) => {
												setLeftBrand(value);
												setLeftModel('');
												setLeftProfileId('');
											}}
										>
											<SelectTrigger>
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
											<Label>Model</Label>
											<Select
												value={leftModel}
												onValueChange={(value) => {
													setLeftModel(value);
													setLeftProfileId('');
												}}
											>
												<SelectTrigger>
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
											<Label>Profile</Label>
											<Select value={leftProfileId} onValueChange={setLeftProfileId}>
												<SelectTrigger>
													<SelectValue placeholder="Select profile" />
												</SelectTrigger>
												<SelectContent>
													{profilesLeft.map((profile) => (
														<SelectItem key={profile.id} value={profile.id}>
															{profile.is_favorite && '⭐ '}
															{profile.lens_model || 'Standard'} -{' '}
															{profile.resolution.width}x{profile.resolution.height}
														</SelectItem>
													))}
												</SelectContent>
											</Select>
										</div>
									)}
								</>
							)}

							{/* Profile Preview */}
							{leftProfileId &&
								(() => {
									const selectedProfile = showLeftFavorites
										? favoriteProfiles.find((p) => p.id === leftProfileId)
										: profilesLeft.find((p) => p.id === leftProfileId);
									return selectedProfile ? (
										<div className="p-3 bg-muted rounded-lg text-sm space-y-1">
											<div className="font-semibold">
												{selectedProfile.camera_brand} {selectedProfile.camera_model}
												{selectedProfile.is_favorite && (
													<span className="ml-2 text-yellow-500">⭐</span>
												)}
											</div>
											<div className="text-muted-foreground">
												{selectedProfile.lens_model || 'Standard'} •{' '}
												{selectedProfile.resolution.width}x{selectedProfile.resolution.height}
											</div>
										</div>
									) : null;
								})()}
						</div>
					</CardContent>
				</Card>

				{/* Right Camera */}
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center gap-2">
							<Video className="h-5 w-5" />
							Right Camera
						</CardTitle>
					</CardHeader>
					<CardContent className="space-y-4">
						{/* Videos */}
						<div className="space-y-3">
							<div className="flex items-center justify-between">
								<Label className="text-base font-semibold">Videos</Label>
								{rightMetadata.filter(Boolean).length > 0 && (
									<div className="text-xs text-muted-foreground">
										{rightMetadata.filter(Boolean).length} video(s) •{' '}
										{calculateTotals(rightMetadata).durationFormatted} •{' '}
										{calculateTotals(rightMetadata).sizeFormatted}
									</div>
								)}
							</div>
							{rightVideoPaths.map((path, index) => (
								<div
									key={index}
									className="space-y-1"
									draggable={!!path}
									onDragStart={() => handleDragStart('right', index)}
									onDragOver={handleDragOver}
									onDrop={() => handleDrop('right', index)}
								>
									<div className="flex gap-2 items-center">
										<GripVertical
											className={`h-4 w-4 ${path ? 'text-muted-foreground cursor-grab' : 'text-muted-foreground/30'}`}
										/>
										<Input
											type="text"
											value={path}
											onChange={(e) => {
												const newPaths = [...rightVideoPaths];
												newPaths[index] = e.target.value;
												setRightVideoPaths(newPaths);
											}}
											placeholder={
												index === rightVideoPaths.length - 1
													? 'Add video...'
													: 'Video file path'
											}
											className="flex-1"
										/>
										<Button
											type="button"
											size="sm"
											onClick={() => handleSelectFile('right', index)}
										>
											Browse
										</Button>
										{path && (
											<Button
												type="button"
												variant="ghost"
												size="sm"
												onClick={() => handleRemoveVideo('right', index)}
											>
												<Trash2 className="h-4 w-4" />
											</Button>
										)}
									</div>
									{rightMetadata[index] && (
										<div className="text-xs text-muted-foreground pl-6 space-x-2">
											<span>{rightMetadata[index].name}</span>
											{rightMetadata[index].duration && (
												<span>• {formatDuration(rightMetadata[index].duration)}</span>
											)}
											<span>• {rightMetadata[index].sizeFormatted}</span>
										</div>
									)}
								</div>
							))}
						</div>

						{/* Profile Selection */}
						<div className="space-y-3 pt-4 border-t">
							<div className="flex items-center justify-between">
								<Label className="text-base font-semibold">
									<Camera className="h-4 w-4 inline mr-1" />
									Lens Profile
								</Label>
								<div className="flex gap-2">
									<Button
										type="button"
										size="sm"
										variant="outline"
										onClick={handleCopyFromLeft}
										disabled={!leftProfileId}
									>
										<ArrowRight className="h-3 w-3 mr-1" />
										Copy Left
									</Button>
									<Button
										type="button"
										size="sm"
										variant={showRightFavorites ? 'default' : 'outline'}
										onClick={() => {
											setShowRightFavorites(!showRightFavorites);
											if (!showRightFavorites) {
												setRightBrand('');
												setRightModel('');
											}
										}}
									>
										<Star className={`h-3 w-3 mr-1 ${showRightFavorites ? 'fill-current' : ''}`} />
										Favorites
									</Button>
								</div>
							</div>

							{showRightFavorites ? (
								<div>
									<Label>Profile</Label>
									<Select value={rightProfileId} onValueChange={setRightProfileId}>
										<SelectTrigger>
											<SelectValue
												placeholder={
													loadingFavorites ? 'Loading...' : 'Select favorite profile'
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
								<>
									<div>
										<Label>Brand</Label>
										<Select
											value={rightBrand}
											onValueChange={(value) => {
												setRightBrand(value);
												setRightModel('');
												setRightProfileId('');
											}}
										>
											<SelectTrigger>
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
											<Label>Model</Label>
											<Select
												value={rightModel}
												onValueChange={(value) => {
													setRightModel(value);
													setRightProfileId('');
												}}
											>
												<SelectTrigger>
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
											<Label>Profile</Label>
											<Select value={rightProfileId} onValueChange={setRightProfileId}>
												<SelectTrigger>
													<SelectValue placeholder="Select profile" />
												</SelectTrigger>
												<SelectContent>
													{profilesRight.map((profile) => (
														<SelectItem key={profile.id} value={profile.id}>
															{profile.is_favorite && '⭐ '}
															{profile.lens_model || 'Standard'} -{' '}
															{profile.resolution.width}x{profile.resolution.height}
														</SelectItem>
													))}
												</SelectContent>
											</Select>
										</div>
									)}
								</>
							)}

							{/* Profile Preview */}
							{rightProfileId &&
								(() => {
									const selectedProfile = showRightFavorites
										? favoriteProfiles.find((p) => p.id === rightProfileId)
										: profilesRight.find((p) => p.id === rightProfileId);
									return selectedProfile ? (
										<div className="p-3 bg-muted rounded-lg text-sm space-y-1">
											<div className="font-semibold">
												{selectedProfile.camera_brand} {selectedProfile.camera_model}
												{selectedProfile.is_favorite && (
													<span className="ml-2 text-yellow-500">⭐</span>
												)}
											</div>
											<div className="text-muted-foreground">
												{selectedProfile.lens_model || 'Standard'} •{' '}
												{selectedProfile.resolution.width}x{selectedProfile.resolution.height}
											</div>
										</div>
									) : null;
								})()}
						</div>
					</CardContent>
				</Card>
			</div>

			{/* Action Buttons */}
			<div className="flex justify-between items-center pt-4">
				<Button type="button" variant="outline" onClick={handleCancel} disabled={isSubmitting}>
					Cancel
				</Button>
				<Button onClick={handleSubmit} disabled={isSubmitting} size="lg" className="px-8">
					{isSubmitting ? 'Creating Match...' : 'Create & Start Processing'}
				</Button>
			</div>
		</div>
	);
}
