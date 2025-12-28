import React, { useState, useEffect } from 'react';
import { validateProfileData } from '../api/profiles';
import { useBrands, useModels } from '../hooks/useProfiles';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';

function generateProfileId(brand, model, lens, width, height) {
	const parts = [
		brand.toLowerCase().replace(/[^a-z0-9]/g, '-'),
		model.toLowerCase().replace(/[^a-z0-9]/g, '-'),
		lens ? lens.toLowerCase().replace(/[^a-z0-9]/g, '-') : null,
		`${width}x${height}`,
	]
		.filter(Boolean)
		.join('-')
		.replace(/-+/g, '-');
	return parts;
}

export default function ProfileForm({ profile, onSubmit, onCancel }) {
	const [formData, setFormData] = useState({
		id: '',
		camera_brand: '',
		camera_model: '',
		lens_model: '',
		resolution: { width: 1920, height: 1080 },
		distortion_model: 'fisheye_kb4',
		camera_matrix: { fx: 1000, fy: 1000, cx: 960, cy: 540 },
		distortion_coeffs: [0, 0, 0, 0],
		calib_dimension: { width: 1920, height: 1080 },
		note: '',
		metadata: {},
	});

	const [errors, setErrors] = useState([]);
	const [isSubmitting, setIsSubmitting] = useState(false);

	// Fetch existing brands and models
	const { brands } = useBrands();
	const { models } = useModels(formData.camera_brand);

	useEffect(() => {
		if (profile) {
			setFormData(profile);
		}
	}, [profile]);

	const handleChange = (field, value) => {
		setFormData((prev) => ({ ...prev, [field]: value }));
	};

	const handleNestedChange = (parent, field, value) => {
		setFormData((prev) => ({
			...prev,
			[parent]: { ...prev[parent], [field]: parseFloat(value) || 0 },
		}));
	};

	const handleArrayChange = (index, value) => {
		setFormData((prev) => {
			const newCoeffs = [...prev.distortion_coeffs];
			newCoeffs[index] = parseFloat(value) || 0;
			return { ...prev, distortion_coeffs: newCoeffs };
		});
	};

	const handleSubmit = async (e) => {
		e.preventDefault();

		// Auto-generate ID from brand, model, lens, and resolution
		const generatedId = generateProfileId(
			formData.camera_brand,
			formData.camera_model,
			formData.lens_model,
			formData.resolution.width,
			formData.resolution.height
		);

		// Add metadata tracking
		const now = new Date().toISOString();
		const metadata = {
			...formData.metadata,
			source: 'user',
			is_custom: true,
		};

		if (profile) {
			// Editing existing profile
			metadata.last_modified = now;
			metadata.modified_by = 'user';
		} else {
			// Creating new profile
			metadata.created_at = now;
			metadata.created_by = 'user';
		}

		const submissionData = {
			...formData,
			id: profile ? profile.id : generatedId,
			metadata,
		};

		const validation = validateProfileData(submissionData);
		if (!validation.valid) {
			setErrors(validation.errors);
			return;
		}

		setErrors([]);
		setIsSubmitting(true);

		try {
			await onSubmit(submissionData);
		} catch (err) {
			setErrors([err.message]);
		} finally {
			setIsSubmitting(false);
		}
	};

	return (
		<form onSubmit={handleSubmit} className="w-full p-4 border rounded bg-white space-y-4">
			<h3 className="text-lg font-bold">{profile ? 'Edit Profile' : 'Create New Profile'}</h3>

			{errors.length > 0 && (
				<Alert variant="destructive">
					<AlertDescription>
						<p className="font-bold mb-2">Validation Errors:</p>
						<ul className="list-disc list-inside text-sm">
							{errors.map((err, i) => (
								<li key={i}>{err}</li>
							))}
						</ul>
					</AlertDescription>
				</Alert>
			)}

			<div className="space-y-4">
				{profile && (
					<div>
						<Label htmlFor="profile-id">Profile ID</Label>
						<Input id="profile-id" type="text" value={profile.id} disabled className="bg-gray-100" />
						<p className="text-xs text-muted-foreground mt-1">ID cannot be changed when editing</p>
					</div>
				)}

				<div className="grid grid-cols-2 gap-4">
					<div>
						<Label htmlFor="camera-brand">
							Camera Brand <span className="text-red-500">*</span>
						</Label>
						<Input
							id="camera-brand"
							type="text"
							list="brands-list"
							value={formData.camera_brand}
							onChange={(e) => handleChange('camera_brand', e.target.value)}
							placeholder="e.g., GoPro"
							required
						/>
						<datalist id="brands-list">
							{brands.map((brand) => (
								<option key={brand} value={brand} />
							))}
						</datalist>
						{brands.length > 0 && (
							<p className="text-xs text-muted-foreground mt-1">Existing brands: {brands.join(', ')}</p>
						)}
					</div>

					<div>
						<Label htmlFor="camera-model">
							Camera Model <span className="text-red-500">*</span>
						</Label>
						<Input
							id="camera-model"
							type="text"
							list="models-list"
							value={formData.camera_model}
							onChange={(e) => handleChange('camera_model', e.target.value)}
							placeholder="e.g., HERO10 Black"
							required
						/>
						<datalist id="models-list">
							{models.map((model) => (
								<option key={model} value={model} />
							))}
						</datalist>
						{models.length > 0 && (
							<p className="text-xs text-muted-foreground mt-1">
								Existing models for {formData.camera_brand}: {models.join(', ')}
							</p>
						)}
					</div>
				</div>

				<div>
					<Label htmlFor="lens-model">Lens Model</Label>
					<Input
						id="lens-model"
						type="text"
						value={formData.lens_model}
						onChange={(e) => handleChange('lens_model', e.target.value)}
						placeholder="e.g., Linear, Wide, Ultrawide"
					/>
				</div>

				<div>
					<Label>
						Resolution <span className="text-red-500">*</span>
					</Label>
					<div className="grid grid-cols-2 gap-2">
						<Input
							type="number"
							value={formData.resolution.width}
							onChange={(e) => handleNestedChange('resolution', 'width', e.target.value)}
							placeholder="Width"
							required
						/>
						<Input
							type="number"
							value={formData.resolution.height}
							onChange={(e) => handleNestedChange('resolution', 'height', e.target.value)}
							placeholder="Height"
							required
						/>
					</div>
				</div>

				<div>
					<Label>
						Camera Matrix <span className="text-red-500">*</span>
					</Label>
					<div className="grid grid-cols-2 gap-2">
						<Input
							type="number"
							step="0.01"
							value={formData.camera_matrix.fx}
							onChange={(e) => handleNestedChange('camera_matrix', 'fx', e.target.value)}
							placeholder="fx (focal length x)"
							required
						/>
						<Input
							type="number"
							step="0.01"
							value={formData.camera_matrix.fy}
							onChange={(e) => handleNestedChange('camera_matrix', 'fy', e.target.value)}
							placeholder="fy (focal length y)"
							required
						/>
						<Input
							type="number"
							step="0.01"
							value={formData.camera_matrix.cx}
							onChange={(e) => handleNestedChange('camera_matrix', 'cx', e.target.value)}
							placeholder="cx (optical center x)"
							required
						/>
						<Input
							type="number"
							step="0.01"
							value={formData.camera_matrix.cy}
							onChange={(e) => handleNestedChange('camera_matrix', 'cy', e.target.value)}
							placeholder="cy (optical center y)"
							required
						/>
					</div>
				</div>

				<div>
					<Label>
						Distortion Coefficients (fisheye_kb4) <span className="text-red-500">*</span>
					</Label>
					<div className="grid grid-cols-4 gap-2">
						{[0, 1, 2, 3].map((i) => (
							<Input
								key={i}
								type="number"
								step="0.0001"
								value={formData.distortion_coeffs[i]}
								onChange={(e) => handleArrayChange(i, e.target.value)}
								placeholder={`k${i + 1}`}
								required
							/>
						))}
					</div>
					<p className="text-xs text-muted-foreground mt-1">4 coefficients required for fisheye_kb4 model</p>
				</div>

				<div>
					<Label>Calibration Dimension (optional)</Label>
					<div className="grid grid-cols-2 gap-2">
						<Input
							type="number"
							value={formData.calib_dimension?.width || ''}
							onChange={(e) => handleNestedChange('calib_dimension', 'width', e.target.value)}
							placeholder="Width"
						/>
						<Input
							type="number"
							value={formData.calib_dimension?.height || ''}
							onChange={(e) => handleNestedChange('calib_dimension', 'height', e.target.value)}
							placeholder="Height"
						/>
					</div>
				</div>

				<div>
					<Label htmlFor="note">Note</Label>
					<Textarea
						id="note"
						value={formData.note}
						onChange={(e) => handleChange('note', e.target.value)}
						placeholder="Additional information about this profile"
						rows={3}
					/>
				</div>
			</div>

			<div className="flex gap-2">
				<Button type="submit" disabled={isSubmitting}>
					{isSubmitting ? 'Saving...' : profile ? 'Update Profile' : 'Create Profile'}
				</Button>
				<Button type="button" variant="outline" onClick={onCancel} disabled={isSubmitting}>
					Cancel
				</Button>
			</div>
		</form>
	);
}
