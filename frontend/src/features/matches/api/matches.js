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

/**
 * Start processing a match (transcoding + calibration)
 */
export async function processMatch(matchId) {
	return api.post(`/matches/${matchId}/process`);
}

/**
 * Start transcoding only (no calibration)
 */
export async function transcodeMatch(matchId) {
	return api.post(`/matches/${matchId}/transcode`);
}

/**
 * Process match with pre-warped frames from frontend
 */
export async function processMatchWithFrames(matchId, leftFrameBlob, rightFrameBlob) {
	const formData = new FormData();
	formData.append('left_frame', leftFrameBlob, 'left_frame.png');
	formData.append('right_frame', rightFrameBlob, 'right_frame.png');

	const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
	const response = await fetch(`${apiBaseUrl}/matches/${matchId}/process-with-frames`, {
		method: 'POST',
		body: formData,
	});

	const text = await response.text();

	if (!response.ok) {
		let errorMessage = 'Failed to process frames';
		try {
			const error = JSON.parse(text);
			errorMessage = error.detail || errorMessage;
		} catch {
			errorMessage = text || errorMessage;
		}
		throw new Error(errorMessage);
	}

	try {
		return JSON.parse(text);
	} catch (e) {
		console.error('Failed to parse response:', text);
		throw new Error('Invalid response from server');
	}
}

/**
 * Get processing status of a match
 */
export async function getMatchStatus(matchId) {
	return api.get(`/matches/${matchId}/status`);
}
