import React, { useState, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { useBrands, useModels, useProfilesByBrandModel } from '../hooks/useProfiles';
import { listFavoriteProfiles, listProfilesMetadata } from '../api/profiles';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Search, Package, Star, X } from 'lucide-react';
import { sortBrands, sortModels, normalizeProfile } from '@/lib/normalize';

const ProfileBrowser = forwardRef(function ProfileBrowser({ onSelect, selectedProfileId }, ref) {
	const [selectedBrand, setSelectedBrand] = useState('');
	const [selectedModel, setSelectedModel] = useState('');
	const [searchQuery, setSearchQuery] = useState('');
	const [globalSearchQuery, setGlobalSearchQuery] = useState('');
	const [globalSearchResults, setGlobalSearchResults] = useState([]);
	const [globalSearchLoading, setGlobalSearchLoading] = useState(false);
	const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
	const [favorites, setFavorites] = useState([]);
	const [favoritesLoading, setFavoritesLoading] = useState(false);
	const [optimisticallyRemoved, setOptimisticallyRemoved] = useState(new Set());

	// Is in global search mode when there's a global search query
	const isGlobalSearchMode = globalSearchQuery.trim().length > 0;

	const { brands: rawBrands, loading: brandsLoading, refetch: refetchBrands } = useBrands();
	const { models: rawModels, loading: modelsLoading, refetch: refetchModels } = useModels(selectedBrand);
	const { profiles: rawProfiles, loading: profilesLoading, refetch: refetchProfiles } = useProfilesByBrandModel(selectedBrand, selectedModel);

	// Refetch all data
	const refetch = useCallback(() => {
		setOptimisticallyRemoved(new Set());
		refetchBrands();
		if (selectedBrand) refetchModels();
		if (selectedBrand && selectedModel) refetchProfiles();
		if (showFavoritesOnly) {
			listFavoriteProfiles()
				.then((data) => setFavorites(data.map(normalizeProfile)))
				.catch((err) => console.error('Failed to reload favorites:', err));
		}
	}, [refetchBrands, refetchModels, refetchProfiles, selectedBrand, selectedModel, showFavoritesOnly]);

	// Remove a profile optimistically
	const removeProfile = useCallback((profileId) => {
		setOptimisticallyRemoved((prev) => new Set([...prev, profileId]));
	}, []);

	// Expose methods to parent via ref
	useImperativeHandle(ref, () => ({
		refetch,
		removeProfile,
	}), [refetch, removeProfile]);

	// Load favorites when favorites mode is enabled
	useEffect(() => {
		if (showFavoritesOnly) {
			setFavoritesLoading(true);
			listFavoriteProfiles()
				.then((data) => {
					setFavorites(data.map(normalizeProfile));
				})
				.catch((err) => {
					console.error('Failed to load favorites:', err);
					setFavorites([]);
				})
				.finally(() => {
					setFavoritesLoading(false);
				});
		}
	}, [showFavoritesOnly]);

	// Global search with debounce
	useEffect(() => {
		if (!globalSearchQuery.trim()) {
			setGlobalSearchResults([]);
			return;
		}

		const timeoutId = setTimeout(async () => {
			setGlobalSearchLoading(true);
			try {
				// Split query into words and search
				const results = await listProfilesMetadata({ search: globalSearchQuery, limit: 100 });
				// Apply fuzzy matching on client side for better results
				const words = globalSearchQuery.toLowerCase().split(/\s+/).filter(Boolean);
				const filtered = results.filter((profile) => {
					const searchable = [
						profile.camera_brand,
						profile.camera_model,
						profile.lens_model,
						profile.id,
						profile.w && profile.h ? `${profile.w}x${profile.h}` : '',
					].filter(Boolean).join(' ').toLowerCase();
					return words.every((word) => searchable.includes(word));
				});
				setGlobalSearchResults(filtered.map(normalizeProfile));
			} catch (err) {
				console.error('Failed to search profiles:', err);
				setGlobalSearchResults([]);
			} finally {
				setGlobalSearchLoading(false);
			}
		}, 300); // 300ms debounce

		return () => clearTimeout(timeoutId);
	}, [globalSearchQuery]);

	// Normalize and sort brands and models
	const brands = sortBrands(rawBrands);
	const models = sortModels(rawModels);
	const profiles = rawProfiles.map(normalizeProfile);

	const handleBrandChange = (brand) => {
		setSelectedBrand(brand);
		setSelectedModel('');
	};

	const handleModelChange = (model) => {
		setSelectedModel(model);
	};

	const handleToggleFavorites = () => {
		setShowFavoritesOnly(!showFavoritesOnly);
		setGlobalSearchQuery(''); // Clear global search when switching modes
		if (!showFavoritesOnly) {
			// Clear brand/model selection when entering favorites mode
			setSelectedBrand('');
			setSelectedModel('');
		}
	};

	const handleClearGlobalSearch = () => {
		setGlobalSearchQuery('');
		setGlobalSearchResults([]);
	};

	// Determine which profiles to display (exclude optimistically removed)
	const displayProfiles = isGlobalSearchMode
		? globalSearchResults.filter((p) => !optimisticallyRemoved.has(p.id))
		: (showFavoritesOnly ? favorites : profiles).filter((p) => !optimisticallyRemoved.has(p.id));

	// Fuzzy search: each word in query must match somewhere in the profile
	const filteredProfiles = displayProfiles
		.filter((profile) => {
			if (!searchQuery) return true;
			// Build searchable text from all profile fields
			const searchable = [
				profile.camera_brand,
				profile.camera_model,
				profile.lens_model,
				profile.id,
				`${profile.resolution.width}x${profile.resolution.height}`,
			].filter(Boolean).join(' ').toLowerCase();
			// Each word in the query must appear somewhere
			const words = searchQuery.toLowerCase().split(/\s+/).filter(Boolean);
			return words.every((word) => searchable.includes(word));
		})
		.sort((a, b) => {
			// Sort favorites to the top
			if (a.is_favorite && !b.is_favorite) return -1;
			if (!a.is_favorite && b.is_favorite) return 1;
			return 0;
		});

	return (
		<div className="w-full">
			{/* Global Search Bar */}
			<div className="mb-4">
				<div className="relative">
					<Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
					<Input
						type="text"
						placeholder="Search all profiles... (e.g., gopro 10 wide)"
						value={globalSearchQuery}
						onChange={(e) => setGlobalSearchQuery(e.target.value)}
						className="pl-9 pr-8"
					/>
					{globalSearchQuery && (
						<Button
							size="icon"
							variant="ghost"
							className="absolute right-1 top-1 h-7 w-7"
							onClick={handleClearGlobalSearch}
						>
							<X className="h-4 w-4" />
						</Button>
					)}
				</div>
				{isGlobalSearchMode && (
					<p className="text-xs text-muted-foreground mt-1">
						{globalSearchLoading ? 'Searching...' : `${displayProfiles.length} results`}
					</p>
				)}
			</div>

			{/* Header with title and favorites toggle */}
			{!isGlobalSearchMode && (
				<div className="flex items-center justify-between mb-4">
					<h3 className="text-lg font-bold">{showFavoritesOnly ? 'Favorites' : 'Browse by Camera'}</h3>
					<div className="flex items-center gap-2">
						<Button
							size="sm"
							variant={showFavoritesOnly ? 'default' : 'outline'}
							onClick={handleToggleFavorites}
						>
							<Star className={`h-4 w-4 mr-1 ${showFavoritesOnly ? 'fill-current' : ''}`} />
							Favorites
						</Button>
						{((showFavoritesOnly && favorites.length > 0) ||
							(selectedBrand && selectedModel && profiles.length > 0)) && (
							<span className="text-xs text-muted-foreground">
								{filteredProfiles.length} of {displayProfiles.length}
							</span>
						)}
					</div>
				</div>
			)}

			<div className="space-y-4">
				{/* Global Search Results */}
				{isGlobalSearchMode && (
					<div>
						{globalSearchLoading ? (
							<div className="flex items-center justify-center p-8 text-muted-foreground">
								<Package className="h-5 w-5 mr-2 animate-pulse" />
								<span>Searching...</span>
							</div>
						) : displayProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No profiles found</p>
								<p className="text-xs text-muted-foreground">Try different search terms</p>
							</div>
						) : (
							<div className="grid gap-1">
								{displayProfiles.slice(0, 50).map((profile) => (
									<Card
										key={profile.id}
										className={`cursor-pointer hover:bg-accent hover:border-primary/50 transition-all py-0 gap-0 ${
											selectedProfileId === profile.id ? 'bg-accent border-primary' : ''
										}`}
										onClick={() => onSelect && onSelect(profile)}
									>
										<CardContent className="px-2.5 py-1">
											<div className="flex items-center justify-between gap-2">
												<div className="text-sm font-semibold flex items-center gap-1.5">
													{profile.is_favorite && <span className="text-yellow-500 text-xs">⭐</span>}
													{profile.camera_brand} {profile.camera_model}
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
								))}
								{displayProfiles.length > 50 && (
									<p className="text-xs text-center text-muted-foreground py-2">
										Showing first 50 of {displayProfiles.length} results
									</p>
								)}
							</div>
						)}
					</div>
				)}

				{/* Browse Mode */}
				{!isGlobalSearchMode && !showFavoritesOnly && (
					<>
						<div>
							<Label htmlFor="brand-select" className="mb-1.5 block">
								Brand
							</Label>
							<Select value={selectedBrand} onValueChange={handleBrandChange} disabled={brandsLoading}>
								<SelectTrigger id="brand-select" className="w-full">
									<SelectValue placeholder="-- select brand --" />
								</SelectTrigger>
								<SelectContent>
									{brands.map((brand) => (
										<SelectItem key={brand} value={brand}>
											{brand}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>

						{selectedBrand && (
							<div>
								<Label htmlFor="model-select" className="mb-1.5 block">
									Model
								</Label>
								<Select
									value={selectedModel}
									onValueChange={handleModelChange}
									disabled={modelsLoading}
								>
									<SelectTrigger id="model-select" className="w-full">
										<SelectValue placeholder="-- select model --" />
									</SelectTrigger>
									<SelectContent>
										{models.map((model) => (
											<SelectItem key={model} value={model}>
												{model}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							</div>
						)}
					</>
				)}

				{/* Favorites Mode */}
				{!isGlobalSearchMode && showFavoritesOnly && (
					<div>
						<div className="flex items-center justify-between mb-2">
							<Label>Favorite Profiles</Label>
							{displayProfiles.length > 3 && (
								<div className="relative flex-1 max-w-xs ml-4">
									<Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
									<Input
										type="text"
										placeholder="Filter favorites..."
										value={searchQuery}
										onChange={(e) => setSearchQuery(e.target.value)}
										className="pl-8 h-9 text-sm"
									/>
								</div>
							)}
						</div>
						{favoritesLoading ? (
							<div className="flex items-center justify-center p-8 text-muted-foreground">
								<Package className="h-5 w-5 mr-2 animate-pulse" />
								<span>Loading favorites...</span>
							</div>
						) : displayProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Star className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No favorites yet</p>
								<p className="text-xs text-muted-foreground">
									Mark profiles as favorites to see them here
								</p>
							</div>
						) : filteredProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No matches found</p>
								<p className="text-xs text-muted-foreground">Try adjusting your filter</p>
							</div>
						) : (
							<div className="grid gap-1 mt-2">
								{filteredProfiles.map((profile) => (
									<Card
										key={profile.id}
										className={`cursor-pointer hover:bg-accent hover:border-primary/50 transition-all py-0 gap-0 ${
											selectedProfileId === profile.id ? 'bg-accent border-primary' : ''
										}`}
										onClick={() => onSelect && onSelect(profile)}
									>
										<CardContent className="px-2.5 py-1">
											<div className="flex items-center justify-between gap-2">
												<div className="text-sm font-semibold flex items-center gap-1.5">
													<span className="text-yellow-500 text-xs">⭐</span>
													{profile.camera_brand} {profile.camera_model}
												</div>
												<div className="text-xs text-muted-foreground">
													{profile.resolution.width}×{profile.resolution.height}
												</div>
											</div>
											{profile.lens_model && (
												<div className="text-xs text-muted-foreground mt-0.5">
													{profile.lens_model}
												</div>
											)}
										</CardContent>
									</Card>
								))}
							</div>
						)}
					</div>
				)}

				{/* Browse Mode - Profile List */}
				{!isGlobalSearchMode && !showFavoritesOnly && selectedBrand && selectedModel && (
					<div>
						<div className="flex items-center justify-between mb-2">
							<Label>Profiles</Label>
							{displayProfiles.length > 3 && (
								<div className="relative flex-1 max-w-xs ml-4">
									<Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
									<Input
										type="text"
										placeholder="Filter profiles..."
										value={searchQuery}
										onChange={(e) => setSearchQuery(e.target.value)}
										className="pl-8 h-9 text-sm"
									/>
								</div>
							)}
						</div>
						{profilesLoading ? (
							<div className="flex items-center justify-center p-8 text-muted-foreground">
								<Package className="h-5 w-5 mr-2 animate-pulse" />
								<span>Loading profiles...</span>
							</div>
						) : displayProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Package className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No profiles found</p>
								<p className="text-xs text-muted-foreground">
									No lens profiles exist for {selectedBrand} {selectedModel}
								</p>
							</div>
						) : filteredProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No matches found</p>
								<p className="text-xs text-muted-foreground">Try adjusting your filter</p>
							</div>
						) : (
							<div className="grid gap-1 mt-2">
								{filteredProfiles.map((profile) => (
									<Card
										key={profile.id}
										className={`cursor-pointer hover:bg-accent hover:border-primary/50 transition-all py-0 gap-0 ${
											selectedProfileId === profile.id ? 'bg-accent border-primary' : ''
										}`}
										onClick={() => onSelect && onSelect(profile)}
									>
										<CardContent className="px-2.5 py-1">
											<div className="flex items-center justify-between gap-2">
												<div className="text-sm font-semibold flex items-center gap-1.5">
													{profile.is_favorite && (
														<span className="text-yellow-500 text-xs">⭐</span>
													)}
													{profile.lens_model || 'Standard Lens'}
												</div>
												<div className="text-xs text-muted-foreground">
													{profile.resolution.width}×{profile.resolution.height}
												</div>
											</div>
										</CardContent>
									</Card>
								))}
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
});

export default ProfileBrowser;
