import { api } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

const HEALTH_URL = '/health';

export async function getHealth() {
	try {
		const data = await api.get(HEALTH_URL);
		return data || { status: 'offline' };
	} catch {
		return { status: 'offline' };
	}
}

/**
 * React Query hook for pinging the server.
 * Options:
 *  - enabled: boolean to enable/disable auto fetch
 */
export function useHealthCheck() {
	return useQuery({
		queryKey: ['health'],
		queryFn: getHealth,
		refetchInterval: 5000,
		retry: true,
	});
}

export default useHealthCheck;
