/**
 * Lens Profile API Client
 *
 * Provides functions to interact with the lens profile management API.
 */

import { api } from '@/lib/api';

/**
 * List all lens profiles
 */
export async function listProfiles() {
	return api.get('/profiles');
}

/**
 * Get a specific lens profile by ID
 */
export async function getProfile(profileId) {
	return api.get(`/profiles/${profileId}`);
}

/**
 * Create a new lens profile
 */
export async function createProfile(profileData) {
	return api.post('/profiles', profileData);
}

/**
 * Update an existing lens profile
 */
export async function updateProfile(profileId, profileData) {
	return api.put(`/profiles/${profileId}`, profileData);
}

/**
 * Delete a lens profile
 */
export async function deleteProfile(profileId) {
	return api.delete(`/profiles/${profileId}`);
}

/**
 * List all camera brands
 */
export async function listBrands() {
	return api.get('/profiles/hierarchy/brands');
}

/**
 * List all models for a specific brand
 */
export async function listModels(brand) {
	return api.get(`/profiles/hierarchy/brands/${encodeURIComponent(brand)}/models`);
}

/**
 * List all profiles for a specific brand and model
 */
export async function listProfilesByBrandModel(brand, model) {
	return api.get(`/profiles/hierarchy/brands/${encodeURIComponent(brand)}/models/${encodeURIComponent(model)}`);
}

/**
 * Get profile hierarchy (brands -> models -> profiles)
 */
export async function getProfileHierarchy() {
	const brands = await listBrands();
	const hierarchy = {};

	for (const brand of brands) {
		const models = await listModels(brand);
		hierarchy[brand] = {};

		for (const model of models) {
			const profiles = await listProfilesByBrandModel(brand, model);
			hierarchy[brand][model] = profiles;
		}
	}

	return hierarchy;
}

/**
 * Search profiles by criteria (client-side filtering)
 */
export async function searchProfiles(criteria = {}) {
	const allProfiles = await listProfiles();

	return allProfiles.filter((profile) => {
		if (criteria.brand && !profile.camera_brand.toLowerCase().includes(criteria.brand.toLowerCase())) {
			return false;
		}

		if (criteria.model && !profile.camera_model.toLowerCase().includes(criteria.model.toLowerCase())) {
			return false;
		}

		if (
			criteria.lens &&
			profile.lens_model &&
			!profile.lens_model.toLowerCase().includes(criteria.lens.toLowerCase())
		) {
			return false;
		}

		if (criteria.minWidth && profile.resolution.width < criteria.minWidth) {
			return false;
		}
		if (criteria.maxWidth && profile.resolution.width > criteria.maxWidth) {
			return false;
		}
		if (criteria.minHeight && profile.resolution.height < criteria.minHeight) {
			return false;
		}
		if (criteria.maxHeight && profile.resolution.height > criteria.maxHeight) {
			return false;
		}

		return true;
	});
}

/**
 * Validate profile data before submission
 */
export function validateProfileData(profileData) {
	const errors = [];

	if (!profileData.id) {
		errors.push('Profile ID is required');
	} else if (!/^[a-z0-9-]{1,100}$/.test(profileData.id)) {
		errors.push('Profile ID must be lowercase alphanumeric with hyphens, max 100 chars');
	}

	if (!profileData.camera_brand) {
		errors.push('Camera brand is required');
	}

	if (!profileData.camera_model) {
		errors.push('Camera model is required');
	}

	if (!profileData.resolution) {
		errors.push('Resolution is required');
	} else {
		if (!profileData.resolution.width || profileData.resolution.width <= 0) {
			errors.push('Resolution width must be positive');
		}
		if (!profileData.resolution.height || profileData.resolution.height <= 0) {
			errors.push('Resolution height must be positive');
		}
	}

	if (profileData.distortion_model !== 'fisheye_kb4') {
		errors.push('Distortion model must be "fisheye_kb4"');
	}

	if (!profileData.camera_matrix) {
		errors.push('Camera matrix is required');
	} else {
		const { fx, fy, cx, cy } = profileData.camera_matrix;
		if (!fx || fx <= 0) errors.push('Camera matrix fx must be positive');
		if (!fy || fy <= 0) errors.push('Camera matrix fy must be positive');
		if (!cx || cx <= 0) errors.push('Camera matrix cx must be positive');
		if (!cy || cy <= 0) errors.push('Camera matrix cy must be positive');
	}

	if (!Array.isArray(profileData.distortion_coeffs) || profileData.distortion_coeffs.length !== 4) {
		errors.push('Distortion coefficients must be an array of exactly 4 numbers');
	}

	return {
		valid: errors.length === 0,
		errors,
	};
}
