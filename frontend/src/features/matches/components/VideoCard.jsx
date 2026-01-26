import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Video } from 'lucide-react';
import VideoList from './VideoList';

/**
 * VideoCard - Card wrapper for a single camera's video list
 * Encapsulates the Card UI pattern for video uploads
 */
export default function VideoCard({ title, side, paths, metadata, handlers }) {
	return (
		<Card>
			<CardHeader>
				<CardTitle className="flex items-center gap-2">
					<Video className="h-5 w-5" />
					{title}
				</CardTitle>
			</CardHeader>
			<CardContent>
				<VideoList
					side={side}
					videoPaths={paths}
					metadata={metadata}
					onSelectFiles={handlers.handleSelectFiles}
					onRemoveVideo={handlers.handleRemoveVideo}
					onDragStart={handlers.handleDragStart}
					onDrop={handlers.handleDrop}
				/>
			</CardContent>
		</Card>
	);
}
