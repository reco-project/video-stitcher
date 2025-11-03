import React from 'react';
import { useHealthCheck } from '../api/health';

export default function Health() {
	const { data, isLoading, isError } = useHealthCheck();
	const healthStatus = data?.status || 'offline';

	if (isLoading) return <div>Loading...</div>;
	if (isError) return <div>Error loading health status</div>;

	return (
		<>
			{healthStatus === 'healthy' ? (
				<p className="text-green-700">Backend is Online</p>
			) : (
				<p className="text-red-700">Backend is Offline</p>
			)}
		</>
	);
}
