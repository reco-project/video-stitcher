import axios from 'axios';
import { env } from '@/config/env';

export const api = axios.create({
	baseURL: env.API_BASE_URL,
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
