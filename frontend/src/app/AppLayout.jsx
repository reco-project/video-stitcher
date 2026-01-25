import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { useMatches } from '@/features/matches/hooks/useMatches';
import { trackTelemetryEvent } from '@/lib/telemetry';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import StatusBar from './components/StatusBar';
import TelemetryConsentDialog from '@/components/TelemetryConsentDialog';

/**
 * AppLayout wraps all pages with persistent header and sidebar navigation
 * This provides consistent context and navigation across the entire app
 */
export default function AppLayout({ children }) {
	const { matches, refetch } = useMatches();
	const location = useLocation();
	const isOnProcessingPage = location.pathname.startsWith('/processing');
	const wasOnProcessingPageRef = React.useRef(isOnProcessingPage);

	// Set up polling interval (separate from state management)
	useEffect(() => {
		const interval = setInterval(() => {
			refetch();
		}, 15000);

		return () => clearInterval(interval);
	}, [refetch]);

	// Telemetry: app open (opt-in)
	useEffect(() => {
		trackTelemetryEvent('app_open');
	}, []);

	// Manage processing state based on matches and location
	useEffect(() => {
		// Skip state management if on processing page - that page handles it
		if (isOnProcessingPage) {
			wasOnProcessingPageRef.current = true;
			return;
		}

		// If we just left the processing page, don't check immediately
		// Let the regular polling (every 15s) handle it when matches refresh
		const justLeftProcessingPage = wasOnProcessingPageRef.current && !isOnProcessingPage;

		if (justLeftProcessingPage) {
			wasOnProcessingPageRef.current = false;
			return; // Skip check, wait for next polling cycle
		}

		// Check if any match is actively processing
		const hasActiveProcessing = matches.some(
			(match) => match.status === 'transcoding' || match.status === 'calibrating'
		);

		// Only set to false when no processing detected
		if (!hasActiveProcessing && window.electronAPI?.setProcessingState) {
			window.electronAPI.setProcessingState(false, 'AppLayout:polling');
		}
	}, [matches, isOnProcessingPage]);

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

			{/* First-run telemetry consent */}
			<TelemetryConsentDialog />
		</div>
	);
}
