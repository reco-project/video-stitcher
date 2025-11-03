const createEnv = (overrides = {}) => {
	return {
		API_BASE_URL: 'http://127.0.0.1:8000/',
		...overrides,
	};
};
export const env = createEnv();
