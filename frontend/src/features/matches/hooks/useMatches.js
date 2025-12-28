/**
 * React hooks for match management
 */

import { useState, useEffect, useCallback } from 'react';
import { listMatches, getMatch, createMatch, updateMatch, deleteMatch } from '../api/matches';

/**
 * Hook to fetch and manage all matches
 */
export function useMatches() {
	const [matches, setMatches] = useState([]);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchMatches = useCallback(async () => {
		try {
			setLoading(true);
			setError(null);
			const data = await listMatches();
			setMatches(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		fetchMatches();
	}, [fetchMatches]);

	return { matches, loading, error, refetch: fetchMatches };
}

/**
 * Hook to fetch a specific match by ID
 */
export function useMatch(matchId) {
	const [match, setMatch] = useState(null);
	const [loading, setLoading] = useState(true);
	const [error, setError] = useState(null);

	const fetchMatch = useCallback(async () => {
		if (!matchId) {
			setMatch(null);
			setLoading(false);
			return;
		}

		try {
			setLoading(true);
			setError(null);
			const data = await getMatch(matchId);
			setMatch(data);
		} catch (err) {
			setError(err.message);
		} finally {
			setLoading(false);
		}
	}, [matchId]);

	useEffect(() => {
		fetchMatch();
	}, [fetchMatch]);

	return { match, loading, error, refetch: fetchMatch };
}

/**
 * Hook for match CRUD operations
 */
export function useMatchMutations() {
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);

	const create = useCallback(async (matchData) => {
		try {
			setLoading(true);
			setError(null);
			const result = await createMatch(matchData);
			return result;
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	const update = useCallback(async (matchId, matchData) => {
		try {
			setLoading(true);
			setError(null);
			const result = await updateMatch(matchId, matchData);
			return result;
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	const remove = useCallback(async (matchId) => {
		try {
			setLoading(true);
			setError(null);
			await deleteMatch(matchId);
		} catch (err) {
			setError(err.message);
			throw err;
		} finally {
			setLoading(false);
		}
	}, []);

	return {
		create,
		update,
		delete: remove,
		loading,
		error,
	};
}
