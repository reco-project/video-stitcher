import React, { useState, useEffect } from 'react';
import { useBrands, useModels, useProfilesByBrandModel } from '../hooks/useProfiles';
import { listFavoriteProfiles } from '../api/profiles';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Search, Package, Star } from 'lucide-react';
import { sortBrands, sortModels, normalizeProfile } from '@/lib/normalize';

export default function ProfileBrowser({ onSelect, selectedProfileId }) {
	const [selectedBrand, setSelectedBrand] = useState('');
	const [selectedModel, setSelectedModel] = useState('');
	const [searchQuery, setSearchQuery] = useState('');
	const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
	const [favorites, setFavorites] = useState([]);
	const [favoritesLoading, setFavoritesLoading] = useState(false);

	const { brands: rawBrands, loading: brandsLoading } = useBrands();
	const { models: rawModels, loading: modelsLoading } = useModels(selectedBrand);
	const { profiles: rawProfiles, loading: profilesLoading } = useProfilesByBrandModel(selectedBrand, selectedModel);

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
		if (!showFavoritesOnly) {
			// Clear brand/model selection when entering favorites mode
			setSelectedBrand('');
			setSelectedModel('');
		}
	};

	// Determine which profiles to display
	const displayProfiles = showFavoritesOnly ? favorites : profiles;

	// Filter profiles by search query
	const filteredProfiles = displayProfiles
		.filter((profile) => {
			if (!searchQuery) return true;
			const query = searchQuery.toLowerCase();
			return (
				profile.lens_model?.toLowerCase().includes(query) ||
				profile.camera_brand?.toLowerCase().includes(query) ||
				profile.camera_model?.toLowerCase().includes(query) ||
				profile.id.toLowerCase().includes(query) ||
				`${profile.resolution.width}x${profile.resolution.height}`.includes(query)
			);
		})
		.sort((a, b) => {
			// Sort favorites to the top
			if (a.is_favorite && !b.is_favorite) return -1;
			if (!a.is_favorite && b.is_favorite) return 1;
			return 0;
		});

	return (
		<div className="w-full">
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

			<div className="space-y-4">
				{!showFavoritesOnly && (
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

				{(showFavoritesOnly || (selectedBrand && selectedModel)) && (
					<div>
						<div className="flex items-center justify-between mb-2">
							<Label>Profiles</Label>
							{displayProfiles.length > 3 && (
								<div className="relative flex-1 max-w-xs ml-4">
									<Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
									<Input
										type="text"
										placeholder="Search profiles..."
										value={searchQuery}
										onChange={(e) => setSearchQuery(e.target.value)}
										className="pl-8 h-9 text-sm"
									/>
								</div>
							)}
						</div>
						{(showFavoritesOnly ? favoritesLoading : profilesLoading) ? (
							<div className="flex items-center justify-center p-8 text-muted-foreground">
								<Package className="h-5 w-5 mr-2 animate-pulse" />
								<span>Loading profiles...</span>
							</div>
						) : displayProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Star className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">
									{showFavoritesOnly ? 'No favorites yet' : 'No profiles found'}
								</p>
								<p className="text-xs text-muted-foreground">
									{showFavoritesOnly
										? 'Mark profiles as favorites to see them here'
										: `No lens profiles exist for ${selectedBrand} ${selectedModel}`}
								</p>
							</div>
						) : filteredProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No matches found</p>
								<p className="text-xs text-muted-foreground">Try adjusting your search query</p>
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
													{showFavoritesOnly
														? `${profile.camera_brand} ${profile.camera_model}`
														: profile.lens_model || 'Standard Lens'}
												</div>
												<div className="text-xs text-muted-foreground">
													{profile.resolution.width}×{profile.resolution.height}
												</div>
											</div>
											{showFavoritesOnly && profile.lens_model && (
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
			</div>
		</div>
	);
}
