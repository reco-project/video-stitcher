import axios from 'axios';

const defaultBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';

// Cache settings to avoid reading file on every request
let cachedBaseUrl = null;
let settingsLoaded = false;

async function getBaseUrl() {
	if (settingsLoaded && cachedBaseUrl) {
		return cachedBaseUrl;
	}

	if (window.electronAPI?.readSettings) {
		try {
			const settings = await window.electronAPI.readSettings();
			cachedBaseUrl = settings.apiBaseUrl || defaultBaseUrl;
			settingsLoaded = true;
			return cachedBaseUrl;
		} catch {
			return defaultBaseUrl;
		}
	}
	return defaultBaseUrl;
}

export const api = axios.create({
	baseURL: defaultBaseUrl,
});

// Update baseURL dynamically from cached settings
api.interceptors.request.use(async (config) => {
	config.baseURL = await getBaseUrl();
	return config;
});

api.interceptors.response.use(
	(response) => {
		return response.data;
	},
	(error) => {
		const message = error.response?.data?.message || error.message;
		error.message = `API Error: ${message}`;
		console.error(error);

		return Promise.reject(error);
	}
);
