/**
 * API client for match management
 */

import { api } from '@/lib/api';

/**
 * Shared helper to extract nested fields from backend response
 * Backend returns: { processing: {...}, transcode: {...} }
 */
function extractNestedFields(raw) {
	const processing = raw.processing || {};
	const transcode = raw.transcode || {};
	return { processing, transcode };
}

/**
 * Normalize backend response to consistent flat structure
 * Handles both match objects and status responses
 */
function normalizeMatch(raw) {
	if (!raw) return null;

	const { processing, transcode } = extractNestedFields(raw);

	return {
		// Core fields
		id: raw.id || null,
		name: raw.name || '',
		src: raw.src || null,
		created_at: raw.created_at || null,

		// Video inputs
		left_videos: raw.left_videos || null,
		right_videos: raw.right_videos || null,

		// Processing fields
		status: processing.status || 'pending',
		processing_step: processing.step || null,
		processing_message: processing.message || null,
		processing_started_at: processing.started_at || null,
		processing_completed_at: processing.completed_at || null,

		// Error fields
		error_message: processing.error_message || null,
		error_code: processing.error_code || null,

		// Transcode fields
		fps: transcode.fps || null,
		transcode_progress: transcode.progress || null,
		transcode_fps: transcode.fps || null,
		transcode_speed: transcode.speed || null,
		transcode_current_time: transcode.current_time || null,
		transcode_total_duration: transcode.total_duration || null,
		transcode_encoder: transcode.encoder || null,

		// Progress (use transcode progress if available)
		progress_percent: transcode.progress || null,

		// Match-specific fields
		left_uniforms: raw.left_uniforms || null,
		right_uniforms: raw.right_uniforms || null,
		params: raw.params || null,
		num_matches: raw.num_matches || null,
		confidence: raw.confidence || null,

		// Metadata and settings
		metadata: raw.metadata || null,
		quality_settings: raw.quality_settings || null,
		viewed: raw.viewed || false,

		// Keep raw for special cases
		_raw: raw,
	};
}

/**
 * List all matches
 */
export async function listMatches() {
	const raw = await api.get('/matches');
	if (!Array.isArray(raw)) return raw;
	return raw.map(normalizeMatch);
}

/**
 * Get a specific match by ID
 */
export async function getMatch(matchId) {
	const raw = await api.get(`/matches/${matchId}`);
	return normalizeMatch(raw);
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
 * Start transcoding (video synchronization and stacking)
 * This is the first step of processing - it prepares videos for frame extraction and calibration
 */
export async function processMatch(matchId) {
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
	} catch {
		console.error('Failed to parse response:', text);
		throw new Error('Invalid response from server');
	}
}

/**
 * Cancel ongoing processing for a match
 */
export async function cancelProcessing(matchId) {
	return api.post(`/matches/${matchId}/cancel`);
}

/**
 * Get processing status of a match
 */
export async function getMatchStatus(matchId) {
	const raw = await api.get(`/matches/${matchId}/status`);
	return normalizeMatch(raw);
}
