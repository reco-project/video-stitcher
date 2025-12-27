import React, { useState } from 'react';
import { useBrands, useModels, useProfilesByBrandModel } from '../hooks/useProfiles';

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

			<div className="grid gap-3">
				<div>
					<label className="block mb-1 font-bold text-sm">Brand</label>
					<select
						className="w-full p-2 rounded border"
						value={selectedBrand}
						onChange={(e) => handleBrandChange(e.target.value)}
						disabled={brandsLoading}
					>
						<option value="">-- select brand --</option>
						{brands.map((brand) => (
							<option key={brand} value={brand}>
								{brand}
							</option>
						))}
					</select>
				</div>

				{selectedBrand && (
					<div>
						<label className="block mb-1 font-bold text-sm">Model</label>
						<select
							className="w-full p-2 rounded border"
							value={selectedModel}
							onChange={(e) => handleModelChange(e.target.value)}
							disabled={modelsLoading}
						>
							<option value="">-- select model --</option>
							{models.map((model) => (
								<option key={model} value={model}>
									{model}
								</option>
							))}
						</select>
					</div>
				)}

				{selectedBrand && selectedModel && (
					<div>
						<label className="block mb-1 font-bold text-sm">Profile</label>
						{profilesLoading ? (
							<div className="text-sm text-gray-500">Loading...</div>
						) : profiles.length === 0 ? (
							<div className="text-sm text-gray-500">No profiles found</div>
						) : (
							<div className="grid gap-2">
								{profiles.map((profile) => (
									<div
										key={profile.id}
										className="p-3 border rounded hover:bg-gray-50 cursor-pointer transition"
										onClick={() => onSelect && onSelect(profile)}
									>
										<div className="font-bold">{profile.lens_model || 'Standard'}</div>
										<div className="text-xs text-gray-500">
											{profile.resolution.width}x{profile.resolution.height}
										</div>
									</div>
								))}
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	);
}
