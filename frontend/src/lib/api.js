import axios from 'axios';
import { getApiBaseUrl } from '@/hooks/useSettings';

export const api = axios.create({
	baseURL: getApiBaseUrl(),
});

// Update baseURL dynamically from settings
api.interceptors.request.use((config) => {
	config.baseURL = getApiBaseUrl();
	return config;
});

api.interceptors.response.use(
	(response) => {
		console.log('API response:', response);
		return response.data;
	},
	(error) => {
		const message = error.response?.data?.message || error.message;
		error.message = `API Error: ${message}`;
		console.error(error);

		return Promise.reject(error);
	}
);
