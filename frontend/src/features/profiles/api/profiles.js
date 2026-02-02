/**
 * Lens Profile API Client
 *
 * Provides functions to interact with the lens profile management API.
 */

import { api } from '@/lib/api';

/**
 * List lens profiles with optional filters (efficient metadata-only response)
 * 
 * @param {Object} options - Query options
 * @param {string} [options.brand] - Filter by camera brand (substring match)
 * @param {string} [options.model] - Filter by camera model (substring match)
 * @param {string} [options.lens] - Filter by lens model (substring match)
 * @param {number} [options.w] - Filter by exact width
 * @param {number} [options.h] - Filter by exact height
 * @param {boolean} [options.official] - Filter by official status
 * @param {string} [options.search] - Full-text search query
 * @param {number} [options.limit] - Maximum results to return
 * @param {number} [options.offset] - Number of results to skip
 * @returns {Promise<Array>} List of profile metadata objects
 */
export async function listProfilesMetadata(options = {}) {
    const params = new URLSearchParams();

    if (options.brand) params.append('brand', options.brand);
    if (options.model) params.append('model', options.model);
    if (options.lens) params.append('lens', options.lens);
    if (options.w !== undefined) params.append('w', options.w);
    if (options.h !== undefined) params.append('h', options.h);
    if (options.official !== undefined) params.append('official', options.official);
    if (options.search) params.append('search', options.search);
    if (options.limit) params.append('limit', options.limit);
    if (options.offset) params.append('offset', options.offset);

    const query = params.toString();
    return api.get(`/profiles/list${query ? '?' + query : ''}`);
}

/**
 * Get count of profiles matching filters
 * 
 * @param {Object} options - Same filter options as listProfilesMetadata
 * @returns {Promise<{count: number}>} Count object
 */
export async function getProfilesCount(options = {}) {
    const params = new URLSearchParams();

    if (options.brand) params.append('brand', options.brand);
    if (options.model) params.append('model', options.model);
    if (options.lens) params.append('lens', options.lens);
    if (options.w !== undefined) params.append('w', options.w);
    if (options.h !== undefined) params.append('h', options.h);
    if (options.official !== undefined) params.append('official', options.official);
    if (options.search) params.append('search', options.search);

    const query = params.toString();
    return api.get(`/profiles/count${query ? '?' + query : ''}`);
}

/**
 * List all lens profiles (full data - use sparingly)
 * 
 * @deprecated Use listProfilesMetadata for efficient listing
 */
export async function listProfiles() {
    return api.get('/profiles');
}

/**
 * Get a specific lens profile by ID (includes full calibration data)
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
 * 
 * For user profiles: Updates in place
 * For official profiles: Auto-creates a user copy with the changes
 * 
 * Check response metadata.source to see if it's 'user' or 'official'
 * Check response metadata.duplicated_from to see if it was auto-duplicated
 * 
 * @param {string} profileId - ID of the profile to update
 * @param {Object} profileData - Updated profile data (must include new ID if changing)
 * @returns {Promise<Object>} Updated or newly created profile
 */
export async function updateProfile(profileId, profileData) {
    return api.put(`/profiles/${profileId}`, profileData);
}

/**
 * Duplicate an existing profile as a new user profile
 * 
 * Use this when you want to explicitly create a copy with a different ID.
 * For editing official profiles with the same ID, just use updateProfile()
 * which will auto-duplicate.
 * 
 * @param {string} profileId - ID of the profile to duplicate
 * @param {string} newId - New unique ID for the duplicated profile
 * @returns {Promise<Object>} The newly created profile
 */
export async function duplicateProfile(profileId, newId) {
    return api.post(`/profiles/${profileId}/duplicate`, { new_id: newId });
}

/**
 * Delete a lens profile (only user profiles can be deleted)
 */
export async function deleteProfile(profileId) {
    return api.delete(`/profiles/${profileId}`);
}

/**
 * Toggle favorite status for a lens profile
 */
export async function toggleFavorite(profileId, isFavorite) {
    return api.patch(`/profiles/${profileId}/favorite`, { is_favorite: isFavorite });
}

/**
 * List IDs of favorite profiles (fast)
 */
export async function listFavoriteIds() {
    return api.get('/profiles/favorites/ids');
}

/**
 * List all favorite profiles
 */
export async function listFavoriteProfiles() {
    return api.get('/profiles/favorites/list');
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
 * Search profiles by criteria (uses efficient server-side filtering)
 */
export async function searchProfiles(criteria = {}) {
    return listProfilesMetadata({
        brand: criteria.brand,
        model: criteria.model,
        lens: criteria.lens,
        w: criteria.minWidth, // Note: exact match only now
        h: criteria.minHeight,
        search: criteria.query,
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
