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
	const [error, setError] = useState(null);

	const handleSelectFile = async (side, index) => {
		try {
			// Check if electron API is available
			if (!window.electronAPI || !window.electronAPI.selectVideoFile) {
				throw new Error('File selection not available. Please run in Electron.');
			}

			const filePath = await window.electronAPI.selectVideoFile();

			if (filePath) {
				if (side === 'left') {
					const newPaths = [...leftVideoPaths];
					newPaths[index] = filePath;
					setLeftVideoPaths(newPaths);
				} else {
					const newPaths = [...rightVideoPaths];
					newPaths[index] = filePath;
					setRightVideoPaths(newPaths);
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

	const handleNext = () => {
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
						<Button type="button" variant="outline" size="sm" onClick={() => handleAddVideo('left')}>
							+ Add Video
						</Button>
					</div>
					{leftVideoPaths.map((path, index) => (
						<div key={index} className="flex gap-2">
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
					))}
				</div>

				<div className="space-y-3">
					<div className="flex justify-between items-center">
						<Label>Right Camera Videos</Label>
						<Button type="button" variant="outline" size="sm" onClick={() => handleAddVideo('right')}>
							+ Add Video
						</Button>
					</div>
					{rightVideoPaths.map((path, index) => (
						<div key={index} className="flex gap-2">
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
					))}
				</div>

				<div className="flex justify-end">
					<Button onClick={handleNext}>Next: Assign Profiles</Button>
				</div>
			</CardContent>
		</Card>
	);
}
