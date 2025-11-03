import React, { Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from 'react-error-boundary';

const queryClient = new QueryClient();

export const AppProvider = ({ children }) => {
	return (
		<Suspense fallback={<div>Loading...</div>}>
			<ErrorBoundary fallback={<div>Something went wrong</div>}>
				<QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
			</ErrorBoundary>
		</Suspense>
	);
};
