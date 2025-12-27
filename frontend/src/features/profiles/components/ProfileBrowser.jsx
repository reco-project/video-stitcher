import React, { useState } from 'react';
import { useBrands, useModels, useProfilesByBrandModel } from '../hooks/useProfiles';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

export default function ProfileBrowser({ onSelect }) {
	const [selectedBrand, setSelectedBrand] = useState('');
	const [selectedModel, setSelectedModel] = useState('');

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

	return (
		<div className="w-full">
			<h3 className="text-lg font-bold mb-2">Browse by Camera</h3>

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
						<Label>Profile</Label>
						{profilesLoading ? (
							<div className="text-sm text-muted-foreground">Loading...</div>
						) : profiles.length === 0 ? (
							<div className="text-sm text-muted-foreground">No profiles found</div>
						) : (
							<div className="grid gap-2 mt-2">
								{profiles.map((profile) => (
									<Card
										key={profile.id}
										className="cursor-pointer hover:bg-accent transition-colors"
										onClick={() => onSelect && onSelect(profile)}
									>
										<CardHeader className="p-3">
											<CardTitle className="text-sm">
												{profile.lens_model || 'Standard'}
											</CardTitle>
										</CardHeader>
										<CardContent className="p-3 pt-0">
											<div className="text-xs text-muted-foreground">
												{profile.resolution.width}x{profile.resolution.height}
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
}
