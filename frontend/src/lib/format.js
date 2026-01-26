/**
 * Format bytes to human readable string
 * @param {number} bytes - Number of bytes
 * @returns {string} Formatted string (e.g., "1.5 GB")
 */
export function formatBytes(bytes) {
	if (bytes === 0) return '0 B';
	const k = 1024;
	const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));
	return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format duration in seconds to human readable string
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted string (e.g., "1h 30m 45s")
 */
export function formatDuration(seconds) {
	if (!seconds) return '0s';
	const hours = Math.floor(seconds / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	const secs = Math.floor(seconds % 60);
	if (hours > 0) return `${hours}h ${minutes}m ${secs}s`;
	if (minutes > 0) return `${minutes}m ${secs}s`;
	return `${secs}s`;
}

/**
 * Calculate totals from metadata array
 * @param {Array} metadata - Array of metadata objects with size and duration
 * @returns {Object} Object with size, sizeFormatted, duration, durationFormatted
 */
export function calculateTotals(metadata) {
	const totalSize = metadata.reduce((sum, m) => sum + (m?.size || 0), 0);
	const totalDuration = metadata.reduce((sum, m) => sum + (m?.duration || 0), 0);
	return {
		size: totalSize,
		sizeFormatted: formatBytes(totalSize),
		duration: totalDuration,
		durationFormatted: formatDuration(totalDuration),
	};
}
