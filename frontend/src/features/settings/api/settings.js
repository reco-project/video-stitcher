/**
 * API client for backend settings
 */

import { env } from '@/config/env';

const API_BASE_URL = env.API_BASE_URL;

/**
 * Get available encoders and current settings
 * @returns {Promise<Object>} Encoder info with available encoders and current selection
 */
export async function getEncoderSettings() {
	const response = await fetch(`${API_BASE_URL}/settings/encoders`);
	if (!response.ok) {
		throw new Error(`Failed to get encoder settings: ${response.statusText}`);
	}
	return response.json();
}

/**
 * Update encoder preference
 * @param {string} encoder - Encoder type (auto, h264_nvenc, h264_qsv, h264_amf, libx264)
 * @returns {Promise<Object>} Update confirmation
 */
export async function updateEncoderSettings(encoder) {
	const response = await fetch(`${API_BASE_URL}/settings/encoders`, {
		method: 'PUT',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({ encoder }),
	});

	if (!response.ok) {
		throw new Error(`Failed to update encoder settings: ${response.statusText}`);
	}
	return response.json();
}

/**
 * Get all settings
 * @returns {Promise<Object>} All settings
 */
export async function getAllSettings() {
	const response = await fetch(`${API_BASE_URL}/settings/`);
	if (!response.ok) {
		throw new Error(`Failed to get settings: ${response.statusText}`);
	}
	return response.json();
}
