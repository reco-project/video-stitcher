/**
 * Hook for managing match processing operations
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { processMatch, getMatchStatus } from '../api/matches';

/**
 * Hook to start and monitor match processing
 * @param {string} matchId - Match ID to process
 * @param {object} options - Configuration options
 * @param {number} options.pollInterval - Status polling interval in ms (default: 2000)
 * @param {boolean} options.autoPoll - Start polling automatically (default: false)
 */
export function useMatchProcessing(matchId, options = {}) {
	const { pollInterval = 2000, autoPoll = false } = options;

	const [status, setStatus] = useState(null);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [isPolling, setIsPolling] = useState(false);
	const pollIntervalRef = useRef(null);

	// Start processing
	const startProcessing = useCallback(async () => {
		if (!matchId) {
			throw new Error('Match ID is required');
		}
		try {
			setLoading(true);
			setError(null);
			const result = await processMatch(matchId);
			setStatus(result);
			setLoading(false);
			return result;
		} catch (err) {
			setError(err.message || 'Failed to start processing');
			setLoading(false);
			throw err;
		}
	}, [matchId]);

	// Fetch current status
	const fetchStatus = useCallback(async () => {
		if (!matchId) {
			console.warn('fetchStatus called without matchId');
			return null;
		}
		try {
			const result = await getMatchStatus(matchId);
			setStatus(result);
			setError(null);
			return result;
		} catch (err) {
			setError(err.message || 'Failed to fetch status');
			throw err;
		}
	}, [matchId]);

	// Stop polling
	const stopPolling = useCallback(() => {
		if (pollIntervalRef.current) {
			clearInterval(pollIntervalRef.current);
			pollIntervalRef.current = null;
		}
		setIsPolling(false);
	}, []);

	// Start polling
	const startPolling = useCallback(() => {
		if (!matchId) {
			console.error('startPolling called without matchId');
			return;
		}

		if (isPolling || pollIntervalRef.current) return;

		setIsPolling(true);

		// Fetch immediately
		fetchStatus();

		// Then poll at interval
		pollIntervalRef.current = setInterval(async () => {
			try {
				const currentStatus = await fetchStatus();

				// Stop polling if processing is complete or errored
				if (currentStatus && (currentStatus.status === 'ready' || currentStatus.status === 'error')) {
					stopPolling();
				}
			} catch (err) {
				console.error('Polling error:', err);
				stopPolling();
			}
		}, pollInterval);
	}, [matchId, isPolling, fetchStatus, pollInterval, stopPolling]); // Auto-poll if enabled and matchId exists
	useEffect(() => {
		if (autoPoll && matchId) {
			startPolling();
		}

		return () => {
			stopPolling();
		};
	}, [autoPoll, matchId]);

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			if (pollIntervalRef.current) {
				clearInterval(pollIntervalRef.current);
				pollIntervalRef.current = null;
			}
		};
	}, []);

	return {
		status,
		loading,
		error,
		isPolling,
		startProcessing,
		fetchStatus,
		startPolling,
		stopPolling,
	};
}
