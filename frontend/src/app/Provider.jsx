import React, { Suspense, useEffect } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from 'react-error-boundary';
import { useDarkMode } from '@/hooks/useDarkMode';

const queryClient = new QueryClient();

function ThemeInitializer({ children }) {
	useDarkMode();

	return children;
}

export const AppProvider = ({ children }) => {
	// Initialize dark mode from localStorage/system preference
	useEffect(() => {
		const saved = localStorage.getItem('theme');
		const isDark = saved ? saved === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;

		if (isDark) {
			document.documentElement.classList.add('dark');
		} else {
			document.documentElement.classList.remove('dark');
		}
	}, []);

	return (
		<Suspense fallback={<div>Loading...</div>}>
			<ErrorBoundary fallback={<div>Something went wrong</div>}>
				<QueryClientProvider client={queryClient}>
					<ThemeInitializer>{children}</ThemeInitializer>
				</QueryClientProvider>
			</ErrorBoundary>
		</Suspense>
	);
};
