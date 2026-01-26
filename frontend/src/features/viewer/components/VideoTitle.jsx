import React from 'react';
import { CheckCircle2, Hash } from 'lucide-react';

/**
 * VideoTitle - Displays the video title with status badges (like YouTube)
 */
export default function VideoTitle({ match }) {
	return (
		<div className="w-full max-w-6xl">
			<div className="flex items-center gap-3 flex-wrap">
				<h1 className="text-xl font-semibold">
					{match.name || match.label}
				</h1>
				<div className="flex items-center gap-2">
					<span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400">
						<CheckCircle2 className="h-3 w-3" />
						Ready
					</span>
					{match.num_matches && (
						<span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">
							<Hash className="h-3 w-3" />
							{match.num_matches} matches
						</span>
					)}
					{match.confidence && (
						<span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
							match.confidence >= 0.8 
								? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
								: match.confidence >= 0.5 
									? 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
									: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
						}`}>
							{(match.confidence * 100).toFixed(0)}%
						</span>
					)}
				</div>
			</div>
		</div>
	);
}
