/**
 * Frontend normalization utilities for camera names and profile data.
 * Provides consistent display formatting without modifying source data.
 */

/**
 * Normalize camera brand for consistent display.
 *
 * @param {string} brand - Camera brand name
 * @returns {string} Normalized brand name
 */
export function normalizeBrand(brand) {
	if (!brand) return brand;

	// GoPro should always be capitalized correctly
	if (brand.toLowerCase() === 'gopro') {
		return 'GoPro';
	}

	// DJI in all caps
	if (brand.toLowerCase() === 'dji') {
		return 'DJI';
	}

	// Default: return as-is
	return brand;
}

/**
 * Normalize camera model for consistent display.
 * Handles GoPro HERO series capitalization and formatting.
 *
 * @param {string} model - Camera model name
 * @returns {string} Normalized model name
 */
export function normalizeModel(model) {
	if (!model) return model;

	// GoPro HERO series: ensure HERO is capitalized
	// Patterns: hero3, Hero10, HERO11, etc.
	model = model.replace(/\bhero(\+?)(\d+)/gi, 'HERO$1$2');

	// Capitalize color/edition names after HERO
	// HERO3 black -> HERO3 Black
	model = model.replace(/\b(black|silver|white|session)\b/gi, (match) => {
		return match.charAt(0).toUpperCase() + match.slice(1).toLowerCase();
	});

	// Clean up spacing around +
	model = model.replace(/\s*\+\s*/, '+');

	// Collapse multiple spaces
	model = model.replace(/\s+/g, ' ');

	return model.trim();
}

/**
 * Sort camera brands in logical order.
 * Common brands first (GoPro, DJI, Sony, etc.), then alphabetically.
 *
 * @param {string[]} brands - Array of brand names
 * @returns {string[]} Sorted array of brand names
 */
export function sortBrands(brands) {
	const priorityBrands = ['GoPro', 'DJI', 'Insta360', 'Sony', 'RED'];

	const normalized = brands.map(normalizeBrand);

	return normalized.sort((a, b) => {
		const aLower = a.toLowerCase();
		const bLower = b.toLowerCase();

		const aPriority = priorityBrands.findIndex((p) => p.toLowerCase() === aLower);
		const bPriority = priorityBrands.findIndex((p) => p.toLowerCase() === bLower);

		// Both in priority list
		if (aPriority !== -1 && bPriority !== -1) {
			return aPriority - bPriority;
		}

		// Only a is priority
		if (aPriority !== -1) return -1;

		// Only b is priority
		if (bPriority !== -1) return 1;

		// Neither in priority, sort alphabetically
		return a.localeCompare(b);
	});
}

/**
 * Sort camera models in logical order.
 * Numeric models first (HERO3, HERO10), then alphabetically.
 *
 * @param {string[]} models - Array of model names
 * @returns {string[]} Sorted array of model names
 */
export function sortModels(models) {
	const normalized = models.map(normalizeModel);

	return normalized.sort((a, b) => {
		// Extract numbers from model names for numeric sorting
		const aMatch = a.match(/(\d+)/);
		const bMatch = b.match(/(\d+)/);

		// Both have numbers
		if (aMatch && bMatch) {
			const aNum = parseInt(aMatch[1], 10);
			const bNum = parseInt(bMatch[1], 10);

			if (aNum !== bNum) {
				return aNum - bNum;
			}
		}

		// Fallback to alphabetical
		return a.localeCompare(b);
	});
}

/**
 * Normalize a complete profile object for display.
 *
 * @param {Object} profile - Profile object
 * @returns {Object} Profile with normalized display fields
 */
export function normalizeProfile(profile) {
	if (!profile) return profile;

	// Handle metadata format (w/h) vs full profile format (resolution.width/height)
	let resolution = profile.resolution;
	if (!resolution && (profile.w || profile.h)) {
		resolution = { width: profile.w, height: profile.h };
	}

	return {
		...profile,
		camera_brand: normalizeBrand(profile.camera_brand),
		camera_model: normalizeModel(profile.camera_model),
		resolution,
	};
}
