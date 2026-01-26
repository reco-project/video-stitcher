import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Video, Trash2, GripVertical } from 'lucide-react';
import { formatDuration, calculateVideoTotals } from '../utils/format';

/**
 * VideoList - Displays and manages a list of video files for one camera side
 * Supports drag-drop reordering, file drops from filesystem, and file selection
 */
export default function VideoList({ side, videoPaths, metadata, onSelectFiles, onRemoveVideo, onDragStart, onDrop }) {
	const [dropTargetIndex, setDropTargetIndex] = useState(null);
	const [isDraggingOverEmpty, setIsDraggingOverEmpty] = useState(false);
	const hasVideos = videoPaths.filter((p) => p).length > 0;

	const handleDragOver = (e) => {
		e.preventDefault();
		e.stopPropagation();
		e.dataTransfer.dropEffect = 'move';
	};

	const handleItemDragOver = (index, e) => {
		setDropTargetIndex(index);
		handleDragOver(e);
	};

	const handleDragLeave = () => {
		setDropTargetIndex(null);
	};

	const handleDrop = (index, e) => {
		setDropTargetIndex(null);
		onDrop(side, index, e);
	};

	const handleEmptyDragOver = (e) => {
		setIsDraggingOverEmpty(true);
		handleDragOver(e);
	};

	const handleEmptyDragLeave = () => {
		setIsDraggingOverEmpty(false);
	};

	const handleEmptyDrop = (e) => {
		setIsDraggingOverEmpty(false);
		onDrop(side, 0, e);
	};

	return (
		<div className="space-y-2 transition-all duration-300">
			{!hasVideos ? (
				<div
					className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300 ${
						isDraggingOverEmpty ? 'border-primary bg-primary/10 scale-[1.02]' : 'hover:bg-accent/50'
					}`}
					onClick={() => onSelectFiles(side)}
					onDragOver={handleEmptyDragOver}
					onDragLeave={handleEmptyDragLeave}
					onDrop={handleEmptyDrop}
				>
					<Video className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
					<p className="text-sm text-muted-foreground">Click to browse or drag and drop video files</p>
					<p className="text-xs text-muted-foreground mt-1">Supports multiple files</p>
				</div>
			) : (
				<>
					<div className="flex items-center justify-between pb-2">
						<div className="text-xs text-muted-foreground">
							{calculateVideoTotals(metadata).count} video(s) • {calculateVideoTotals(metadata).durationFormatted} •{' '}
							{calculateVideoTotals(metadata).sizeFormatted}
						</div>
						<Button type="button" size="sm" onClick={() => onSelectFiles(side)}>
							Add More
						</Button>
					</div>
					<div className="space-y-2">
						{videoPaths.map((path, index) => {
							if (!path) return null;
							const isDropTarget = dropTargetIndex === index;
							return (
								<div
									key={index}
									className="space-y-1 relative transition-all duration-200"
									draggable={!!path}
									onDragStart={(e) => onDragStart(side, index, e)}
									onDragOver={(e) => handleItemDragOver(index, e)}
									onDragLeave={handleDragLeave}
									onDrop={(e) => handleDrop(index, e)}
								>
									{isDropTarget && (
										<div className="absolute -top-1 left-0 right-0 h-0.5 bg-primary z-10" />
									)}
									<div className="flex gap-2 items-center">
										<GripVertical className="h-4 w-4 text-muted-foreground cursor-grab" />
										<div className="flex-1 text-sm truncate">{path.split(/[\\/]/).pop()}</div>
										<Button
											type="button"
											variant="ghost"
											size="sm"
											onClick={() => onRemoveVideo(side, index)}
										>
											<Trash2 className="h-4 w-4" />
										</Button>
									</div>
									{metadata[index] && (
										<div className="text-xs text-muted-foreground pl-6 space-x-2">
											{metadata[index].resolution && (
												<span>{metadata[index].resolution}</span>
											)}
											{metadata[index].duration && (
												<span>• {formatDuration(metadata[index].duration)}</span>
											)}
											<span>• {metadata[index].sizeFormatted}</span>
											{metadata[index].createdFormatted && (
												<span>• {metadata[index].createdFormatted}</span>
											)}
										</div>
									)}
								</div>
							);
						})}
						{/* Drop zone at the bottom */}
						<div
							className="relative h-2"
							onDragOver={(e) => {
								setDropTargetIndex(videoPaths.filter((p) => p).length);
								handleDragOver(e);
							}}
							onDragLeave={handleDragLeave}
							onDrop={(e) => handleDrop(videoPaths.filter((p) => p).length, e)}
						>
							{dropTargetIndex === videoPaths.filter((p) => p).length && (
								<div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
							)}
						</div>
					</div>
				</>
			)}
		</div>
	);
}
