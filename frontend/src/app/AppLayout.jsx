import React from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import StatusBar from '@/components/StatusBar';

/**
 * AppLayout wraps all pages with persistent header and sidebar navigation
 * This provides consistent context and navigation across the entire app
 */
export default function AppLayout({ children }) {
	return (
		<div className="flex flex-col h-screen bg-background text-foreground">
			{/* Header */}
			<Header />

			{/* Main Content Area */}
			<main className="flex-1 overflow-y-auto pb-20">{children}</main>

			{/* Footer with Navigation and Status */}
			<Sidebar />

			{/* Status Bar (always visible) */}
			<StatusBar />
		</div>
	);
}
