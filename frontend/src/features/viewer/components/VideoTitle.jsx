import React from 'react';

/**
 * VideoTitle - Displays the video title
 */
export default function VideoTitle({ match }) {
	return (
		<div className="w-full max-w-6xl">
			<h1 className="text-xl font-semibold">{match.name || match.label}</h1>
		</div>
	);
}
