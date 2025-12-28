import React from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Header shows app title with current page indicator
 */
export default function Header() {
	const location = useLocation();

	// Get current page name
	const getPageName = () => {
		if (location.pathname === '/') return 'Home';
		if (location.pathname === '/create') return 'Create Match';
		if (location.pathname.startsWith('/viewer')) return 'Viewer';
		if (location.pathname === '/profiles') return 'Settings';
		return 'Home';
	};

	const pageName = getPageName();

	return (
		<header className="border-b bg-background sticky top-0 z-40">
			<div className="flex items-center justify-between px-6 py-5">
				{/* Left: Project Title */}
				<div className="text-sm font-medium text-muted-foreground">reco-project/video-stitcher</div>

				{/* Center: Current Page */}
				<div className="absolute left-1/2 -translate-x-1/2">
					<h1 className="text-2xl font-bold tracking-tight">{pageName}</h1>
				</div>

				{/* Right: Empty space for balance */}
				<div className="w-20" />
			</div>
		</header>
	);
}
