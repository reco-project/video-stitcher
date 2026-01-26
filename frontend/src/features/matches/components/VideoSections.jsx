import React, { useEffect } from 'react';
import VideoCard from './VideoCard';
import { useVideoManager } from '../hooks/useVideoManager';

/**
 * VideoSections - Manages left and right camera video uploads
 * Encapsulates useVideoManager hook and exposes video paths via onChange
 */
export default function VideoSections({
	initialLeftPaths = [],
	initialRightPaths = [],
	onChange,
}) {
	const { left, right, handlers } = useVideoManager(initialLeftPaths, initialRightPaths);

	// Notify parent when video paths change
	useEffect(() => {
		if (onChange) {
			onChange({ left, right });
		}
	}, [left.paths, right.paths, onChange]);

	// Listen for external file drops from Electron
	useEffect(() => {
		const handleExternalDrop = (event) => {
			if (event.data?.type === 'external-file-drop' && event.data.files?.length > 0) {
				// Default to left side for external drops, user can reorder/move after
				handlers.handleDrop('left', 0, null, event.data.files);
			}
		};

		window.addEventListener('message', handleExternalDrop);
		return () => window.removeEventListener('message', handleExternalDrop);
	}, [handlers]);

	return (
		<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<VideoCard
				title="Left Camera"
				side="left"
				paths={left.paths}
				metadata={left.metadata}
				handlers={handlers}
			/>
			<VideoCard
				title="Right Camera"
				side="right"
				paths={right.paths}
				metadata={right.metadata}
				handlers={handlers}
			/>
		</div>
	);
}
