import { useState, useEffect } from 'react';

const defaultSettings = {
	debugMode: true,
	apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api',
	encoderPreference: 'auto',
	disableHardwareAcceleration: false,
	telemetryEnabled: false,
	telemetryIncludeSystemInfo: false,
	telemetryEndpointUrl: 'https://telemetry.reco-project.org/telemetry',
	telemetryAutoUpload: false,
	telemetryPromptShown: false,
	// Recording settings
	recordingBitrate: 16, // Mbps
	recordingFormat: 'webm',
};

/**
 * Hook to manage application-wide settings
 * Settings are persisted to userData/settings.json via IPC
 */
export function useSettings() {
	const [settings, setSettings] = useState(defaultSettings);
	const [loaded, setLoaded] = useState(false);

	useEffect(() => {
		if (window.electronAPI?.readSettings) {
			window.electronAPI.readSettings().then((s) => {
				setSettings(s);
				setLoaded(true);
			});
		} else {
			setLoaded(true);
		}
	}, []);

	useEffect(() => {
		if (loaded && window.electronAPI?.writeSettings) {
			window.electronAPI.writeSettings(settings);
		}
	}, [settings, loaded]);

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
		loading: !loaded,
	};
}
