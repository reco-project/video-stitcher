/**
 * Helper functions for reading match data.
 * Note: These work with normalized match objects from the API.
 */

/**
 * Get processing status from match data
 */
export function getProcessingStatus(match) {
	return match?.status || 'pending';
}

/**
 * Get processing timestamps
 */
export function getProcessingTimes(match) {
	return {
		startedAt: match?.processing_started_at,
		completedAt: match?.processing_completed_at,
	};
}

/**
 * Get transcode metrics
 */
export function getTranscodeMetrics(match) {
	return {
		fps: match?.fps,
		speed: match?.transcode_speed,
		progress: match?.transcode_progress,
		currentTime: match?.transcode_current_time,
		totalDuration: match?.transcode_total_duration,
		offsetSeconds: match?._raw?.transcode?.offset_seconds,
	};
}

/**
 * Get quality settings
 */
export function getQualitySettings(match) {
	return match?._raw?.quality_settings;
}

/**
 * Calculate processing duration in seconds
 */
export function getProcessingDuration(match) {
	const { startedAt, completedAt } = getProcessingTimes(match);
	if (!startedAt || !completedAt) return null;

	try {
		const duration = (new Date(completedAt) - new Date(startedAt)) / 1000;
		return duration > 0 ? duration : null;
	} catch (e) {
		return null;
	}
}
