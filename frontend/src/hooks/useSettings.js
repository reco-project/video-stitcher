import { useState, useEffect } from 'react';

const SETTINGS_KEY = 'app-settings';

const defaultSettings = {
	debugMode: false,
	apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api',
};

/**
 * Hook to manage application-wide settings
 * Settings are persisted to localStorage
 */
export function useSettings() {
	const [settings, setSettings] = useState(() => {
		try {
			const saved = localStorage.getItem(SETTINGS_KEY);
			return saved ? { ...defaultSettings, ...JSON.parse(saved) } : defaultSettings;
		} catch {
			return defaultSettings;
		}
	});

	useEffect(() => {
		try {
			localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
		} catch (err) {
			console.warn('Failed to save settings:', err);
		}
	}, [settings]);

	const updateSetting = (key, value) => {
		setSettings((prev) => ({ ...prev, [key]: value }));
	};

	const resetSettings = () => {
		setSettings(defaultSettings);
	};

	return {
		settings,
		updateSetting,
		resetSettings,
	};
}

/**
 * Get current API base URL from settings
 */
export function getApiBaseUrl() {
	try {
		const saved = localStorage.getItem(SETTINGS_KEY);
		if (saved) {
			const settings = JSON.parse(saved);
			return settings.apiBaseUrl || defaultSettings.apiBaseUrl;
		}
	} catch {
		// Ignore
	}
	return defaultSettings.apiBaseUrl;
}

/**
 * Check if debug mode is enabled
 */
export function isDebugMode() {
	try {
		const saved = localStorage.getItem(SETTINGS_KEY);
		if (saved) {
			const settings = JSON.parse(saved);
			return settings.debugMode || false;
		}
	} catch {
		// Ignore
	}
	return false;
}
