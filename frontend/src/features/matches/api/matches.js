/**
 * API client for match management
 */

import { api } from '@/lib/api';

/**
 * List all matches
 */
export async function listMatches() {
	return api.get('/matches');
}

/**
 * Get a specific match by ID
 */
export async function getMatch(matchId) {
	return api.get(`/matches/${matchId}`);
}

/**
 * Create a new match
 */
export async function createMatch(matchData) {
	return api.post('/matches', matchData);
}

/**
 * Update an existing match
 */
export async function updateMatch(matchId, matchData) {
	return api.put(`/matches/${matchId}`, matchData);
}

/**
 * Delete a match
 */
export async function deleteMatch(matchId) {
	return api.delete(`/matches/${matchId}`);
}
