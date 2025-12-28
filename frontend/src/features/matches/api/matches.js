/**
 * API client for match management
 */

import { api } from '@/lib/api';

function normalizeMatch(raw) {
	if (!raw) return null;
	return {
		id: raw.id || raw.match_id || raw._id || null,
		name: raw.name || raw.label || raw.title || '',
		src: raw.src || raw.video || raw.source || null,
		status: raw.status || raw.state || raw.phase || 'pending',
		processing_step: raw.processing_step || raw.step || raw.stage || raw.processingStage || null,
		processing_message: raw.processing_message || raw.message || raw.detail || null,
		progress_percent:
			typeof raw.progress_percent === 'number'
				? raw.progress_percent
				: typeof raw.processing_percent === 'number'
					? raw.processing_percent
					: typeof raw.progress === 'number'
						? raw.progress
						: null,
		fps: raw.fps || (raw.metrics && raw.metrics.fps) || null,
		frames_processed:
			raw.frames_processed || (raw.metrics && raw.metrics.frames_processed) || raw.processed_frames || null,
		frames_total: raw.frames_total || (raw.metrics && raw.metrics.frames_total) || raw.total_frames || null,
		audio_sync: raw.audio_sync || (raw.metrics && raw.metrics.audio_sync) || raw.audio || null,
		processing_started_at: raw.processing_started_at || raw.started_at || raw.start_time || null,
		processing_completed_at: raw.processing_completed_at || raw.completed_at || raw.end_time || null,
		error_message: raw.error_message || raw.error || raw.detail || null,
		error_code: raw.error_code || raw.code || null,
		left_uniforms: raw.left_uniforms || null,
		right_uniforms: raw.right_uniforms || null,
		params: raw.params || null,
		num_matches: raw.num_matches || null,
		confidence: raw.confidence || null,
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
 * Start transcoding only (no calibration)
 * @deprecated Use processMatch instead - same endpoint
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
	// Fetch raw status and normalize common backend alias fields so the
	// frontend can rely on a consistent shape.
	const raw = await api.get(`/matches/${matchId}/status`);

	const normalized = {
		// canonical fields
		status: raw.status || raw.state || raw.phase || 'pending',
		processing_step: raw.processing_step || raw.step || raw.stage || raw.processingStage || null,
		processing_message: raw.processing_message || raw.message || raw.detail || null,

		// progress hints (percent)
		progress_percent:
			typeof raw.progress_percent === 'number'
				? raw.progress_percent
				: typeof raw.processing_percent === 'number'
					? raw.processing_percent
					: typeof raw.progress === 'number'
						? raw.progress
						: typeof raw.step_progress === 'number'
							? raw.step_progress
							: null,

		// fps and frame counters
		fps: raw.fps || (raw.metrics && raw.metrics.fps) || raw.transcode_fps || null,
		frames_processed:
			raw.frames_processed || (raw.metrics && raw.metrics.frames_processed) || raw.processed_frames || null,
		frames_total: raw.frames_total || (raw.metrics && raw.metrics.frames_total) || raw.total_frames || null,

		// audio sync info (may be a string or object)
		audio_sync: raw.audio_sync || (raw.metrics && raw.metrics.audio_sync) || raw.audio || null,

		// timestamps
		processing_started_at: raw.processing_started_at || raw.started_at || raw.start_time || null,
		processing_completed_at: raw.processing_completed_at || raw.completed_at || raw.end_time || null,

		// error info
		error_message: raw.error_message || raw.error || raw.detail || null,
		error_code: raw.error_code || raw.code || null,

		// include original payload for any extra fields
		_raw: raw,
	};

	return normalized;
}
