import React, { useState, useEffect, useMemo } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Check, ChevronsUpDown, Star, ChevronRight, ChevronLeft, Loader2 } from 'lucide-react';
import { cn } from '@/lib/cn';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/toast';
import {
	Command,
	CommandEmpty,
	CommandGroup,
	CommandInput,
	CommandItem,
	CommandList,
	CommandSeparator,
} from '@/components/ui/command';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import {
	listProfiles,
	listFavoriteProfiles,
	listBrands,
	listModels,
	listProfilesByBrandModel,
	getProfile,
	toggleFavorite,
} from '@/features/profiles/api/profiles';
import { sortBrands, sortModels } from '@/lib/normalize';

// Simple in-memory cache
const cache = {
	allProfiles: null,
	favorites: null,
	brands: null,
	models: {}, // keyed by brand
	profiles: {}, // keyed by brand:model
	timestamp: {},
};

const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

function isCacheValid(key) {
	return cache.timestamp[key] && Date.now() - cache.timestamp[key] < CACHE_TTL;
}

async function getCachedAllProfiles() {
	if (cache.allProfiles && isCacheValid('allProfiles')) {
		return cache.allProfiles;
	}
	const profiles = await listProfiles();
	cache.allProfiles = profiles;
	cache.timestamp.allProfiles = Date.now();
	return profiles;
}

async function getCachedFavorites() {
	if (cache.favorites && isCacheValid('favorites')) {
		return cache.favorites;
	}
	const favorites = await listFavoriteProfiles();
	cache.favorites = favorites;
	cache.timestamp.favorites = Date.now();
	return favorites;
}

async function getCachedBrands() {
	if (cache.brands && isCacheValid('brands')) {
		return cache.brands;
	}
	const brands = await listBrands();
	cache.brands = sortBrands(brands);
	cache.timestamp.brands = Date.now();
	return cache.brands;
}

async function getCachedModels(brand) {
	const key = `models:${brand}`;
	if (cache.models[brand] && isCacheValid(key)) {
		return cache.models[brand];
	}
	const models = await listModels(brand);
	cache.models[brand] = sortModels(models);
	cache.timestamp[key] = Date.now();
	return cache.models[brand];
}

async function getCachedProfiles(brand, model) {
	const key = `profiles:${brand}:${model}`;
	if (cache.profiles[`${brand}:${model}`] && isCacheValid(key)) {
		return cache.profiles[`${brand}:${model}`];
	}
	const profiles = await listProfilesByBrandModel(brand, model);
	// Sort by lens_model
	const sorted = [...profiles].sort((a, b) => (a.lens_model || 'Standard').localeCompare(b.lens_model || 'Standard'));
	cache.profiles[`${brand}:${model}`] = sorted;
	cache.timestamp[key] = Date.now();
	return sorted;
}

// Clear favorites cache (call when favorite status changes)
export function clearFavoritesCache() {
	cache.favorites = null;
	delete cache.timestamp.favorites;
}

function ProfileComboboxInner({ value, onChange, disabled, className, labelPrefix = '' }) {
	const [open, setOpen] = useState(false);
	const [allProfiles, setAllProfiles] = useState([]);
	const [favorites, setFavorites] = useState([]);
	const [brands, setBrands] = useState([]);
	const [models, setModels] = useState([]);
	const [profiles, setProfiles] = useState([]);
	const [loading, setLoading] = useState(false);
	const [searchQuery, setSearchQuery] = useState('');
	const { showToast } = useToast();

	// Navigation state for browsing: null = root, string = selected brand, [brand, model] for profile list
	const [navBrand, setNavBrand] = useState(null);
	const [navModel, setNavModel] = useState(null);

	// Selected profile details for display
	const [selectedProfile, setSelectedProfile] = useState(null);
	const [loadingSelectedProfile, setLoadingSelectedProfile] = useState(false);

	// Determine if we're in search mode (any text in search box)
	const isSearchMode = searchQuery.trim().length > 0;

	// Optimistically load cached profile info from localStorage
	useEffect(() => {
		if (value) {
			// Try to get cached profile info first for immediate display
			const cachedKey = `profile_display_${value}`;
			const cached = localStorage.getItem(cachedKey);
			if (cached) {
				try {
					const cachedProfile = JSON.parse(cached);
					setSelectedProfile(cachedProfile);
				} catch {
					// Invalid cache, ignore
				}
			}
		}
	}, [value]);

	// Load selected profile details
	useEffect(() => {
		if (value) {
			setLoadingSelectedProfile(true);
			getProfile(value)
				.then((profile) => {
					setSelectedProfile(profile);
					// Cache minimal display info for next time
					if (profile) {
						const cachedKey = `profile_display_${value}`;
						const displayInfo = {
							id: profile.id,
							camera_brand: profile.camera_brand,
							camera_model: profile.camera_model,
							lens_model: profile.lens_model,
							resolution: profile.resolution,
							is_favorite: profile.is_favorite,
						};
						localStorage.setItem(cachedKey, JSON.stringify(displayInfo));
					}
				})
				.catch(() => setSelectedProfile(null))
				.finally(() => setLoadingSelectedProfile(false));
		} else {
			setSelectedProfile(null);
			setLoadingSelectedProfile(false);
		}
	}, [value]);

	// Load data when popover opens
	useEffect(() => {
		if (open) {
			loadInitialData();
		}
	}, [open]);

	// Load models when brand is selected (browse mode)
	useEffect(() => {
		if (navBrand && !navModel && !isSearchMode) {
			loadModels(navBrand);
		}
	}, [navBrand, isSearchMode]);

	// Load profiles when model is selected (browse mode)
	useEffect(() => {
		if (navBrand && navModel && !isSearchMode) {
			loadProfiles(navBrand, navModel);
		}
	}, [navBrand, navModel, isSearchMode]);

	const loadInitialData = async () => {
		setLoading(true);
		try {
			const [allProfilesList, favs, brandList] = await Promise.all([
				getCachedAllProfiles(),
				getCachedFavorites(),
				getCachedBrands(),
			]);
			setAllProfiles(allProfilesList);
			setFavorites(
				favs.sort((a, b) => {
					const brandCompare = (a.camera_brand || '').localeCompare(b.camera_brand || '');
					if (brandCompare !== 0) return brandCompare;
					return (a.camera_model || '').localeCompare(b.camera_model || '');
				})
			);
			setBrands(brandList);
		} catch (err) {
			console.error('Failed to load profile data:', err);
		} finally {
			setLoading(false);
		}
	};

	const loadModels = async (brand) => {
		setLoading(true);
		try {
			const modelList = await getCachedModels(brand);
			setModels(modelList);
		} catch (err) {
			console.error('Failed to load models:', err);
		} finally {
			setLoading(false);
		}
	};

	const loadProfiles = async (brand, model) => {
		setLoading(true);
		try {
			const profileList = await getCachedProfiles(brand, model);
			setProfiles(profileList);
		} catch (err) {
			console.error('Failed to load profiles:', err);
		} finally {
			setLoading(false);
		}
	};

	const handleSelect = (profileId) => {
		onChange(profileId);
		setOpen(false);
		resetNav();
	};

	const handleToggleFavorite = async (e, profile) => {
		e.stopPropagation(); // Prevent selecting the profile

		try {
			const newFavoriteStatus = !profile.is_favorite;
			await toggleFavorite(profile.id, newFavoriteStatus);

			// Update the profile in all relevant state arrays
			const updateProfile = (p) => (p.id === profile.id ? { ...p, is_favorite: newFavoriteStatus } : p);

			setAllProfiles((prev) => prev.map(updateProfile));
			setProfiles((prev) => prev.map(updateProfile));

			if (newFavoriteStatus) {
				// Add to favorites
				setFavorites((prev) =>
					[...prev, { ...profile, is_favorite: true }].sort((a, b) => {
						const brandCompare = (a.camera_brand || '').localeCompare(b.camera_brand || '');
						if (brandCompare !== 0) return brandCompare;
						return (a.camera_model || '').localeCompare(b.camera_model || '');
					})
				);
				showToast({ message: 'Added to favorites', type: 'success' });
			} else {
				// Remove from favorites
				setFavorites((prev) => prev.filter((p) => p.id !== profile.id));
				showToast({ message: 'Removed from favorites', type: 'success' });
			}

			// Update selectedProfile if it's the one being toggled
			if (selectedProfile?.id === profile.id) {
				setSelectedProfile((prev) => ({ ...prev, is_favorite: newFavoriteStatus }));
			}

			// Clear cache to ensure fresh data on next load
			clearFavoritesCache();
			cache.allProfiles = null;
			delete cache.timestamp.allProfiles;
		} catch (err) {
			console.error('Failed to toggle favorite:', err);
			showToast({ message: 'Failed to update favorite', type: 'error' });
		}
	};

	const resetNav = () => {
		setNavBrand(null);
		setNavModel(null);
		setSearchQuery('');
	};

	const goBack = () => {
		if (navModel) {
			setNavModel(null);
			setProfiles([]);
		} else if (navBrand) {
			setNavBrand(null);
			setModels([]);
		}
	};

	// Search results: filter all profiles by query
	const searchResults = useMemo(() => {
		if (!isSearchMode) return [];
		const q = searchQuery.toLowerCase();
		return allProfiles
			.filter(
				(p) =>
					p.camera_brand?.toLowerCase().includes(q) ||
					p.camera_model?.toLowerCase().includes(q) ||
					p.lens_model?.toLowerCase().includes(q) ||
					`${p.camera_brand} ${p.camera_model}`.toLowerCase().includes(q) ||
					`${p.camera_brand} ${p.camera_model} ${p.lens_model || ''}`.toLowerCase().includes(q)
			)
			.sort((a, b) => {
				// Sort favorites first, then alphabetically
				if (a.is_favorite && !b.is_favorite) return -1;
				if (!a.is_favorite && b.is_favorite) return 1;
				const brandCompare = (a.camera_brand || '').localeCompare(b.camera_brand || '');
				if (brandCompare !== 0) return brandCompare;
				const modelCompare = (a.camera_model || '').localeCompare(b.camera_model || '');
				if (modelCompare !== 0) return modelCompare;
				return (a.lens_model || '').localeCompare(b.lens_model || '');
			});
	}, [allProfiles, searchQuery, isSearchMode]);

	// Format profile for display
	const formatProfile = (profile, showBrandModel = true) => {
		if (showBrandModel) {
			return `${profile.camera_brand} ${profile.camera_model} - ${profile.lens_model || 'Standard'} - ${profile.resolution?.width}x${profile.resolution?.height}`;
		}
		return `${profile.lens_model || 'Standard'} - ${profile.resolution?.width}x${profile.resolution?.height}`;
	};

	// Display value
	const displayValue = selectedProfile ? formatProfile(selectedProfile) : 'Select profile...';

	return (
		<Popover open={open} onOpenChange={setOpen}>
			<PopoverTrigger asChild>
				<Button
					variant="outline"
					role="combobox"
					aria-expanded={open}
					disabled={disabled}
					className={cn('!flex justify-start font-normal w-full h-9 px-3 gap-2', className)}
					title={selectedProfile ? displayValue : undefined}
				>
					{labelPrefix && (
						<>
							<span className="shrink-0 text-sm font-semibold">
								{labelPrefix.replace(':', '').trim()}
							</span>
							{selectedProfile && (
								<>
									<div className="w-px h-4 bg-border shrink-0" />
									<span
										onClick={(e) => {
											e.stopPropagation();
											handleToggleFavorite(e, selectedProfile);
										}}
										className="shrink-0 hover:scale-110 transition-transform cursor-pointer"
										title={
											selectedProfile.is_favorite ? 'Remove from favorites' : 'Add to favorites'
										}
									>
										<Star
											className={cn(
												'h-4 w-4',
												selectedProfile.is_favorite
													? 'fill-yellow-400 text-yellow-400'
													: 'text-muted-foreground'
											)}
										/>
									</span>
								</>
							)}
							<div className="w-px h-4 bg-border shrink-0" />
						</>
					)}
					{!labelPrefix && selectedProfile && (
						<span
							onClick={(e) => {
								e.stopPropagation();
								handleToggleFavorite(e, selectedProfile);
							}}
							className="shrink-0 hover:scale-110 transition-transform -ml-1 cursor-pointer"
							title={selectedProfile.is_favorite ? 'Remove from favorites' : 'Add to favorites'}
						>
							<Star
								className={cn(
									'h-4 w-4',
									selectedProfile.is_favorite
										? 'fill-yellow-400 text-yellow-400'
										: 'text-muted-foreground'
								)}
							/>
						</span>
					)}
					<span className="truncate flex-1">{displayValue}</span>
					{loadingSelectedProfile && (
						<Loader2 className="h-3 w-3 animate-spin text-muted-foreground shrink-0" />
					)}
					<ChevronsUpDown className="h-4 w-4 shrink-0 opacity-50" />
				</Button>
			</PopoverTrigger>
			<PopoverContent className="w-[400px] p-0" align="start">
				<Command shouldFilter={false}>
					<CommandInput placeholder="Search profiles..." value={searchQuery} onValueChange={setSearchQuery} />
					<CommandList>
						{loading && <div className="py-6 text-center text-sm text-muted-foreground">Loading...</div>}

						{/* SEARCH MODE: Show direct profile results */}
						{!loading && isSearchMode && (
							<>
								{searchResults.length > 0 ? (
									<CommandGroup heading={`Results for ${searchQuery}`}>
										{searchResults.slice(0, 50).map((profile) => (
											<CommandItem
												key={profile.id}
												value={profile.id}
												onSelect={() => handleSelect(profile.id)}
												className="cursor-pointer min-w-0"
											>
												{profile.is_favorite && (
													<Star className="mr-2 h-3 w-3 fill-yellow-400 text-yellow-400 shrink-0" />
												)}
												<span className="truncate flex-1 min-w-0">
													{formatProfile(profile)}
												</span>
												{value === profile.id && <Check className="ml-2 h-4 w-4 shrink-0" />}
											</CommandItem>
										))}
										{searchResults.length > 50 && (
											<div className="py-2 text-center text-xs text-muted-foreground">
												Showing first 50 of {searchResults.length} results
											</div>
										)}
									</CommandGroup>
								) : (
									<CommandEmpty>No profiles found for {searchQuery}</CommandEmpty>
								)}
							</>
						)}

						{/* BROWSE MODE: Favorites + Brand navigation */}
						{!loading && !isSearchMode && !navBrand && (
							<>
								{/* Favorites Section */}
								{favorites.length > 0 && (
									<CommandGroup heading="Favorites">
										{favorites.map((profile) => (
											<CommandItem
												key={profile.id}
												value={profile.id}
												onSelect={() => handleSelect(profile.id)}
												className="cursor-pointer min-w-0"
											>
												<span className="truncate flex-1 min-w-0">
													{formatProfile(profile)}
												</span>
												{value === profile.id && <Check className="ml-2 h-4 w-4 shrink-0" />}
											</CommandItem>
										))}
									</CommandGroup>
								)}

								{favorites.length > 0 && brands.length > 0 && <CommandSeparator />}

								{/* Browse by Brand */}
								<CommandGroup heading="Browse">
									{brands.map((brand) => (
										<CommandItem
											key={brand}
											value={brand}
											onSelect={() => setNavBrand(brand)}
											className="cursor-pointer"
										>
											<span className="flex-1">{brand}</span>
											<ChevronRight className="h-4 w-4 opacity-50" />
										</CommandItem>
									))}
								</CommandGroup>
							</>
						)}

						{/* BROWSE MODE: Model Selection */}
						{!loading && !isSearchMode && navBrand && !navModel && (
							<>
								<CommandGroup>
									<CommandItem onSelect={goBack} className="cursor-pointer text-muted-foreground">
										<ChevronLeft className="mr-2 h-4 w-4" />
										Back to brands
									</CommandItem>
								</CommandGroup>
								<CommandSeparator />
								<CommandGroup heading={navBrand}>
									{models.map((model) => (
										<CommandItem
											key={model}
											value={model}
											onSelect={() => setNavModel(model)}
											className="cursor-pointer"
										>
											<span className="flex-1">{model}</span>
											<ChevronRight className="h-4 w-4 opacity-50" />
										</CommandItem>
									))}
								</CommandGroup>
							</>
						)}

						{/* BROWSE MODE: Profile Selection */}
						{!loading && !isSearchMode && navBrand && navModel && (
							<>
								<CommandGroup>
									<CommandItem onSelect={goBack} className="cursor-pointer text-muted-foreground">
										<ChevronLeft className="mr-2 h-4 w-4" />
										Back to {navBrand}
									</CommandItem>
								</CommandGroup>
								<CommandSeparator />
								<CommandGroup heading={`${navBrand} ${navModel}`}>
									{profiles.map((profile) => (
										<CommandItem
											key={profile.id}
											value={profile.id}
											onSelect={() => handleSelect(profile.id)}
											className="cursor-pointer min-w-0"
										>
											{profile.is_favorite && (
												<Star className="mr-2 h-3 w-3 fill-yellow-400 text-yellow-400 shrink-0" />
											)}
											<span className="truncate flex-1 min-w-0">
												{formatProfile(profile, false)}
											</span>
											{value === profile.id && <Check className="ml-2 h-4 w-4 shrink-0" />}
										</CommandItem>
									))}
								</CommandGroup>
							</>
						)}
					</CommandList>
				</Command>
			</PopoverContent>
		</Popover>
	);
}

export default function ProfileCombobox(props) {
	return (
		<ErrorBoundary
			fallback={<div className="text-destructive text-sm p-2">Something went wrong loading profiles.</div>}
		>
			<ProfileComboboxInner {...props} />
		</ErrorBoundary>
	);
}
