/**
 * React hooks for lens profile management
 */

import { useState, useEffect, useCallback } from 'react';
import {
	listProfiles,
	getProfile,
	createProfile,
	updateProfile,
	deleteProfile,
	listBrands,
	listModels,
	listProfilesByBrandModel,
	searchProfiles,
} from '../api/profiles';

/**
 * Hook to fetch and manage all lens profiles
 */
export function useProfiles() {
	const [profiles, setProfiles] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchProfiles = useCallback(async () => {
		try {
			setLoading(true);
			setError(null);
			const data = await listProfiles();
			setProfiles(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchProfiles();
	}, [fetchProfiles]);

	return { profiles, loading, error, refetch: fetchProfiles };
}

/**
 * Hook to fetch a specific profile by ID
 */
export function useProfile(profileId) {
	const [profile, setProfile] = useState(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchProfile = useCallback(async () => {
		if (!profileId) return;

		try {
			setLoading(true);
			setError(null);
			const data = await getProfile(profileId);
			setProfile(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, [profileId]);

	useEffect(() => {
		fetchProfile();
	}, [fetchProfile]);

	return { profile, loading, error, refetch: fetchProfile };
}

/**
 * Hook for profile CRUD operations
 */
export function useProfileMutations() {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);

	const create = useCallback(async (profileData) => {
		try {
			setLoading(true);
			setError(null);
			const result = await createProfile(profileData);
			return result;
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	const update = useCallback(async (profileId, profileData) => {
		try {
			setLoading(true);
			setError(null);
			const result = await updateProfile(profileId, profileData);
			return result;
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	const remove = useCallback(async (profileId) => {
		try {
			setLoading(true);
			setError(null);
			await deleteProfile(profileId);
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	return { create, update, delete: remove, loading, error };
}

/**
 * Hook to fetch camera brands
 */
export function useBrands() {
	const [brands, setBrands] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchBrands = useCallback(async () => {
		try {
			setLoading(true);
			setError(null);
			const data = await listBrands();
			setBrands(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchBrands();
	}, [fetchBrands]);

	return { brands, loading, error, refetch: fetchBrands };
}

/**
 * Hook to fetch models for a specific brand
 */
export function useModels(brand) {
	const [models, setModels] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchModels = useCallback(async () => {
		if (!brand) {
			setModels([]);
			setLoading(false);
			return;
		}

		try {
			setLoading(true);
			setError(null);
			const data = await listModels(brand);
			setModels(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, [brand]);

	useEffect(() => {
		fetchModels();
	}, [fetchModels]);

	return { models, loading, error, refetch: fetchModels };
}

/**
 * Hook to fetch profiles by brand and model
 */
export function useProfilesByBrandModel(brand, model) {
	const [profiles, setProfiles] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchProfiles = useCallback(async () => {
		if (!brand || !model) {
			setProfiles([]);
			setLoading(false);
			return;
		}

		try {
			setLoading(true);
			setError(null);
			const data = await listProfilesByBrandModel(brand, model);
			setProfiles(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, [brand, model]);

	useEffect(() => {
		fetchProfiles();
	}, [fetchProfiles]);

	return { profiles, loading, error, refetch: fetchProfiles };
}

/**
 * Hook for searching profiles with criteria
 */
export function useProfileSearch(initialCriteria = {}) {
	const [profiles, setProfiles] = useState([]);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [criteria, setCriteria] = useState(initialCriteria);

	const search = useCallback(
		async (newCriteria) => {
			const searchCriteria = newCriteria || criteria;

			try {
				setLoading(true);
				setError(null);
				const data = await searchProfiles(searchCriteria);
				setProfiles(data);
				if (newCriteria) {
					setCriteria(newCriteria);
				}
			} catch (err) {
				setError(err.message);
			} finally {
				setLoading(false);
			}
		},
		[criteria]
	);

	useEffect(() => {
		if (Object.keys(criteria).length > 0) {
			search();
		}
	}, []);

	return { profiles, loading, error, search, criteria };
}
