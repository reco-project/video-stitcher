import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

export default function VideoImportStep({ onNext, initialData }) {
	const [name, setName] = useState(initialData?.name || '');
	const [leftVideoPaths, setLeftVideoPaths] = useState(initialData?.left_videos?.map((v) => v.path) || ['']);
	const [rightVideoPaths, setRightVideoPaths] = useState(initialData?.right_videos?.map((v) => v.path) || ['']);
	const [leftMetadata, setLeftMetadata] = useState([]);
	const [rightMetadata, setRightMetadata] = useState([]);
	const [error, setError] = useState(null);
	const [draggedIndex, setDraggedIndex] = useState(null);
	const [draggedSide, setDraggedSide] = useState(null);

	const loadMetadata = async (filePath, side, index) => {
		if (!filePath || !window.electronAPI?.getFileMetadata) return;

		try {
			const metadata = await window.electronAPI.getFileMetadata(filePath);
			if (metadata) {
				if (side === 'left') {
					setLeftMetadata((prev) => {
						const newMeta = [...prev];
						newMeta[index] = metadata;
						return newMeta;
					});
				} else {
					setRightMetadata((prev) => {
						const newMeta = [...prev];
						newMeta[index] = metadata;
						return newMeta;
					});
				}
			}
		} catch (err) {
			console.warn('Failed to load metadata:', err);
		}
	};

	const handleSelectFile = async (side, index) => {
		try {
			if (!window.electronAPI || !window.electronAPI.selectVideoFile) {
				throw new Error('File selection not available. Please run in Electron.');
			}

			const filePath = await window.electronAPI.selectVideoFile();

			if (filePath) {
				if (side === 'left') {
					const newPaths = [...leftVideoPaths];
					newPaths[index] = filePath;
					setLeftVideoPaths(newPaths);
					loadMetadata(filePath, side, index);
				} else {
					const newPaths = [...rightVideoPaths];
					newPaths[index] = filePath;
					setRightVideoPaths(newPaths);
					loadMetadata(filePath, side, index);
				}
				setError(null);
			}
		} catch (err) {
			setError(err.message);
		}
	};

	const handleAddVideo = (side) => {
		if (side === 'left') {
			setLeftVideoPaths([...leftVideoPaths, '']);
		} else {
			setRightVideoPaths([...rightVideoPaths, '']);
		}
	};

	const handleAddMultipleVideos = async (side) => {
		try {
			if (!window.electronAPI || !window.electronAPI.selectVideoFiles) {
				throw new Error('Multi-select not available. Please run in Electron.');
			}

			const filePaths = await window.electronAPI.selectVideoFiles();

			if (filePaths && filePaths.length > 0) {
				if (side === 'left') {
					const newPaths = [...leftVideoPaths.filter((p) => p.trim()), ...filePaths];
					setLeftVideoPaths(newPaths);
					filePaths.forEach((path, idx) => {
						loadMetadata(path, side, leftVideoPaths.length + idx);
					});
				} else {
					const newPaths = [...rightVideoPaths.filter((p) => p.trim()), ...filePaths];
					setRightVideoPaths(newPaths);
					filePaths.forEach((path, idx) => {
						loadMetadata(path, side, rightVideoPaths.length + idx);
					});
				}
				setError(null);
			}
		} catch (err) {
			setError(err.message);
		}
	};

	const handleRemoveVideo = (side, index) => {
		if (side === 'left' && leftVideoPaths.length > 1) {
			setLeftVideoPaths(leftVideoPaths.filter((_, i) => i !== index));
		} else if (side === 'right' && rightVideoPaths.length > 1) {
			setRightVideoPaths(rightVideoPaths.filter((_, i) => i !== index));
		}
	};

	const handlePathChange = (side, index, value) => {
		if (side === 'left') {
			const newPaths = [...leftVideoPaths];
			newPaths[index] = value;
			setLeftVideoPaths(newPaths);
		} else {
			const newPaths = [...rightVideoPaths];
			newPaths[index] = value;
			setRightVideoPaths(newPaths);
		}
	};

	const handleDragStart = (side, index) => {
		setDraggedIndex(index);
		setDraggedSide(side);
	};

	const handleDragOver = (e) => {
		e.preventDefault();
	};

	const handleDrop = (side, dropIndex) => {
		if (draggedIndex === null || draggedSide !== side) return;

		const paths = side === 'left' ? [...leftVideoPaths] : [...rightVideoPaths];
		const metadata = side === 'left' ? [...leftMetadata] : [...rightMetadata];

		const [movedPath] = paths.splice(draggedIndex, 1);
		paths.splice(dropIndex, 0, movedPath);

		const [movedMeta] = metadata.splice(draggedIndex, 1);
		metadata.splice(dropIndex, 0, movedMeta);

		if (side === 'left') {
			setLeftVideoPaths(paths);
			setLeftMetadata(metadata);
		} else {
			setRightVideoPaths(paths);
			setRightMetadata(metadata);
		}

		setDraggedIndex(null);
		setDraggedSide(null);
	};

	const handleMoveUp = (side, index) => {
		if (index === 0) return;
		setDraggedIndex(index);
		setDraggedSide(side);
		handleDrop(side, index - 1);
	};

	const handleMoveDown = (side, index) => {
		const maxIndex = (side === 'left' ? leftVideoPaths : rightVideoPaths).length - 1;
		if (index === maxIndex) return;
		setDraggedIndex(index);
		setDraggedSide(side);
		handleDrop(side, index + 1);
	};

	const handleNext = async () => {
		if (!name.trim()) {
			setError('Please enter a match name');
			return;
		}

		const validLeftPaths = leftVideoPaths.filter((p) => p.trim());
		const validRightPaths = rightVideoPaths.filter((p) => p.trim());

		if (validLeftPaths.length === 0 || validRightPaths.length === 0) {
			setError('Please select at least one video for both left and right cameras');
			return;
		}

		if (window.electronAPI && window.electronAPI.fileExists) {
			try {
				const allPaths = [...validLeftPaths, ...validRightPaths];
				const existenceChecks = await Promise.all(allPaths.map((path) => window.electronAPI.fileExists(path)));

				const missingFiles = allPaths.filter((_, idx) => !existenceChecks[idx]);

				if (missingFiles.length > 0) {
					setError(`The following files do not exist:\n${missingFiles.join('\n')}`);
					return;
				}
			} catch (err) {
				console.warn('Failed to validate file existence:', err);
			}
		}

		onNext({
			name: name.trim(),
			left_videos: validLeftPaths.map((path) => ({ path })),
			right_videos: validRightPaths.map((path) => ({ path })),
		});
	};

	return (
		<Card>
			<CardHeader>
				<CardTitle>Step 1: Import Videos</CardTitle>
				<p className="text-sm text-muted-foreground mt-2">
					Select raw video files from left and right cameras. These will be processed by the backend to create
					a stitched output video.
				</p>
			</CardHeader>
			<CardContent className="space-y-4">
				{error && (
					<Alert variant="destructive">
						<AlertDescription>{error}</AlertDescription>
					</Alert>
				)}

				<div>
					<Label htmlFor="match-name">Match Name</Label>
					<Input
						id="match-name"
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						placeholder="e.g., Match 2024-12-27"
					/>
				</div>

				<div className="space-y-3">
					<div className="flex justify-between items-center">
						<Label>Left Camera Videos</Label>
						<div className="flex gap-2">
							<Button
								type="button"
								variant="outline"
								size="sm"
								onClick={() => handleAddMultipleVideos('left')}
							>
								+ Add Multiple
							</Button>
							<Button type="button" variant="outline" size="sm" onClick={() => handleAddVideo('left')}>
								+ Add One
							</Button>
						</div>
					</div>
					{leftVideoPaths.map((path, index) => (
						<div
							key={index}
							className="space-y-2"
							draggable
							onDragStart={() => handleDragStart('left', index)}
							onDragOver={handleDragOver}
							onDrop={() => handleDrop('left', index)}
						>
							<div className="flex gap-2 items-center">
								<div className="flex flex-col gap-1">
									<Button
										type="button"
										variant="ghost"
										size="sm"
										className="h-6 w-6 p-0"
										onClick={() => handleMoveUp('left', index)}
										disabled={index === 0}
									>
										↑
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="sm"
										className="h-6 w-6 p-0"
										onClick={() => handleMoveDown('left', index)}
										disabled={index === leftVideoPaths.length - 1}
									>
										↓
									</Button>
								</div>
								<Input
									type="text"
									value={path}
									onChange={(e) => handlePathChange('left', index, e.target.value)}
									placeholder="Video file path"
									className="flex-1"
								/>
								<Button type="button" onClick={() => handleSelectFile('left', index)}>
									Browse...
								</Button>
								{leftVideoPaths.length > 1 && (
									<Button
										type="button"
										variant="destructive"
										size="sm"
										onClick={() => handleRemoveVideo('left', index)}
									>
										Remove
									</Button>
								)}
							</div>
							{leftMetadata[index] && (
								<div className="text-xs text-muted-foreground pl-2">
									{leftMetadata[index].name} • {leftMetadata[index].sizeFormatted}
								</div>
							)}
						</div>
					))}
				</div>

				<div className="space-y-3">
					<div className="flex justify-between items-center">
						<Label>Right Camera Videos</Label>
						<div className="flex gap-2">
							<Button
								type="button"
								variant="outline"
								size="sm"
								onClick={() => handleAddMultipleVideos('right')}
							>
								+ Add Multiple
							</Button>
							<Button type="button" variant="outline" size="sm" onClick={() => handleAddVideo('right')}>
								+ Add One
							</Button>
						</div>
					</div>
					{rightVideoPaths.map((path, index) => (
						<div
							key={index}
							className="space-y-2"
							draggable
							onDragStart={() => handleDragStart('right', index)}
							onDragOver={handleDragOver}
							onDrop={() => handleDrop('right', index)}
						>
							<div className="flex gap-2 items-center">
								<div className="flex flex-col gap-1">
									<Button
										type="button"
										variant="ghost"
										size="sm"
										className="h-6 w-6 p-0"
										onClick={() => handleMoveUp('right', index)}
										disabled={index === 0}
									>
										↑
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="sm"
										className="h-6 w-6 p-0"
										onClick={() => handleMoveDown('right', index)}
										disabled={index === rightVideoPaths.length - 1}
									>
										↓
									</Button>
								</div>
								<Input
									type="text"
									value={path}
									onChange={(e) => handlePathChange('right', index, e.target.value)}
									placeholder="Video file path"
									className="flex-1"
								/>
								<Button type="button" onClick={() => handleSelectFile('right', index)}>
									Browse...
								</Button>
								{rightVideoPaths.length > 1 && (
									<Button
										type="button"
										variant="destructive"
										size="sm"
										onClick={() => handleRemoveVideo('right', index)}
									>
										Remove
									</Button>
								)}
							</div>
							{rightMetadata[index] && (
								<div className="text-xs text-muted-foreground pl-2">
									{rightMetadata[index].name} • {rightMetadata[index].sizeFormatted}
								</div>
							)}
						</div>
					))}
				</div>

				<div className="flex justify-end">
					<Button onClick={handleNext}>Next: Assign Profiles</Button>
				</div>
			</CardContent>
		</Card>
	);
}
