import React, { useState } from 'react';
import { useBrands, useModels, useProfilesByBrandModel } from '../hooks/useProfiles';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Search, Camera, Package } from 'lucide-react';

export default function ProfileBrowser({ onSelect }) {
	const [selectedBrand, setSelectedBrand] = useState('');
	const [selectedModel, setSelectedModel] = useState('');
	const [searchQuery, setSearchQuery] = useState('');

	const { brands, loading: brandsLoading } = useBrands();
	const { models, loading: modelsLoading } = useModels(selectedBrand);
	const { profiles, loading: profilesLoading } = useProfilesByBrandModel(selectedBrand, selectedModel);

	const handleBrandChange = (brand) => {
		setSelectedBrand(brand);
		setSelectedModel('');
	};

	const handleModelChange = (model) => {
		setSelectedModel(model);
	};

	// Filter profiles by search query
	const filteredProfiles = profiles.filter((profile) => {
		if (!searchQuery) return true;
		const query = searchQuery.toLowerCase();
		return (
			profile.lens_model?.toLowerCase().includes(query) ||
			profile.id.toLowerCase().includes(query) ||
			`${profile.resolution.width}x${profile.resolution.height}`.includes(query)
		);
	});

	return (
		<div className="w-full">
			<div className="flex items-center justify-between mb-4">
				<h3 className="text-lg font-bold">Browse by Camera</h3>
				{selectedBrand && selectedModel && profiles.length > 0 && (
					<span className="text-xs text-muted-foreground">
						{filteredProfiles.length} of {profiles.length} profiles
					</span>
				)}
			</div>

			<div className="space-y-4">
				<div>
					<Label htmlFor="brand-select">Brand</Label>
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
						<Label htmlFor="model-select">Model</Label>
						<Select value={selectedModel} onValueChange={handleModelChange} disabled={modelsLoading}>
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

				{selectedBrand && selectedModel && (
					<div>
						<div className="flex items-center justify-between mb-2">
							<Label>Profiles</Label>
							{profiles.length > 3 && (
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
						{profilesLoading ? (
							<div className="flex items-center justify-center p-8 text-muted-foreground">
								<Package className="h-5 w-5 mr-2 animate-pulse" />
								<span>Loading profiles...</span>
							</div>
						) : profiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Camera className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No profiles found</p>
								<p className="text-xs text-muted-foreground">
									No lens profiles exist for {selectedBrand} {selectedModel}
								</p>
							</div>
						) : filteredProfiles.length === 0 ? (
							<div className="border-2 border-dashed rounded-lg p-8 text-center">
								<Search className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
								<p className="text-sm font-medium mb-1">No matches found</p>
								<p className="text-xs text-muted-foreground">Try adjusting your search query</p>
							</div>
						) : (
							<div className="grid gap-2 mt-2">
								{filteredProfiles.map((profile) => (
									<Card
										key={profile.id}
										className="cursor-pointer hover:bg-accent hover:border-primary/50 transition-all"
										onClick={() => onSelect && onSelect(profile)}
									>
										<CardHeader className="p-3">
											<CardTitle className="text-sm">
												{profile.lens_model || 'Standard Lens'}
											</CardTitle>
											<div className="text-xs text-muted-foreground mt-1">
												{profile.resolution.width}×{profile.resolution.height}
											</div>
										</CardHeader>
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
