import { useEffect, useState } from 'react';

/**
 * Dark mode hook - manages dark/light theme
 * Persists preference to localStorage
 */
export function useDarkMode() {
	const [isDark, setIsDark] = useState(() => {
		// Check localStorage first
		const saved = localStorage.getItem('theme');
		if (saved) {
			return saved === 'dark';
		}
		// Check system preference
		return window.matchMedia('(prefers-color-scheme: dark)').matches;
	});

	useEffect(() => {
		// Update localStorage
		localStorage.setItem('theme', isDark ? 'dark' : 'light');

		// Update document class
		const html = document.documentElement;
		if (isDark) {
			html.classList.add('dark');
		} else {
			html.classList.remove('dark');
		}
	}, [isDark]);

	const toggle = () => setIsDark((prev) => !prev);

	return { isDark, toggle };
}
