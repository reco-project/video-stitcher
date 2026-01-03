/**
 * Prototype component to test requestVideoFrameCallback dual-video sync.
 *
 * This is a minimal test to validate rVFC synchronization works
 * before fully integrating into the main viewer.
 */

import React, { useEffect, useRef, useState } from 'react';
import { useDualVideoTextures } from '../hooks/useDualVideoTextures.js';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { isRVFCSupported } from '../utils/videoSync.js';

const DualVideoTest = ({ match }) => {
	const [isPlaying, setIsPlaying] = useState(false);

	// For prototype, use top-level or _raw fields for video paths
	const leftVideoSrc = match?.left_video_path || match?._raw?.left_video_path || null;
	const rightVideoSrc = match?.right_video_path || match?._raw?.right_video_path || null;
	const offsetSeconds = match?.audio_offset || match?._raw?.audio_offset || 0;

	// Debug logging
	console.log('DualVideoTest - Match data:', match);
	console.log('DualVideoTest - Left path:', leftVideoSrc);
	console.log('DualVideoTest - Right path:', rightVideoSrc);

	const { leftTexture, rightTexture, videoElement, isReady } = useDualVideoTextures(
		leftVideoSrc,
		rightVideoSrc,
		offsetSeconds,
		{ autoPlay: false }
	);

	const canvasLeftRef = useRef(null);
	const canvasRightRef = useRef(null);

	// Render textures to canvas for visualization
	useEffect(() => {
		if (!leftTexture || !rightTexture || !isReady) return;

		const drawFrame = () => {
			// Draw left texture
			if (canvasLeftRef.current && leftTexture.image) {
				const ctx = canvasLeftRef.current.getContext('2d');
				ctx.drawImage(leftTexture.image, 0, 0, canvasLeftRef.current.width, canvasLeftRef.current.height);
			}

			// Draw right texture
			if (canvasRightRef.current && rightTexture.image) {
				const ctx = canvasRightRef.current.getContext('2d');
				ctx.drawImage(rightTexture.image, 0, 0, canvasRightRef.current.width, canvasRightRef.current.height);
			}

			requestAnimationFrame(drawFrame);
		};

		drawFrame();
	}, [leftTexture, rightTexture, isReady]);

	const handlePlayPause = () => {
		if (videoElement) {
			if (isPlaying) {
				videoElement.pause();
			} else {
				videoElement.play();
			}
			setIsPlaying(!isPlaying);
		}
	};

	if (!leftVideoSrc || !rightVideoSrc) {
		return (
			<Card className="w-full">
				<CardHeader>
					<CardTitle>Dual Video Sync Test</CardTitle>
					<CardDescription>
						This prototype requires separate video paths. Current match uses stacked video format.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<p className="text-sm text-muted-foreground">
						To test rVFC sync, the backend needs to provide separate <code>left_video_path</code> and{' '}
						<code>right_video_path</code> instead of a single stacked video.
					</p>
				</CardContent>
			</Card>
		);
	}

	return (
		<div className="w-full space-y-4">
			<Card>
				<CardHeader>
					<div className="flex items-center justify-between">
						<div>
							<CardTitle>Dual Video Sync Test (rVFC)</CardTitle>
							<CardDescription>Testing requestVideoFrameCallback synchronization</CardDescription>
						</div>
						<Badge variant={isRVFCSupported() ? 'default' : 'destructive'}>
							{isRVFCSupported() ? 'rVFC Supported' : 'rVFC Not Supported (RAF fallback)'}
						</Badge>
					</div>
				</CardHeader>
				<CardContent className="space-y-4">
					{/* Video canvases */}
					<div className="grid grid-cols-2 gap-4">
						<div>
							<p className="text-sm font-medium mb-2">Left Camera</p>
							<canvas
								ref={canvasLeftRef}
								width={640}
								height={360}
								className="w-full border rounded bg-black"
							/>
						</div>
						<div>
							<p className="text-sm font-medium mb-2">Right Camera</p>
							<canvas
								ref={canvasRightRef}
								width={640}
								height={360}
								className="w-full border rounded bg-black"
							/>
						</div>
					</div>

					{/* Controls */}
					<div className="flex items-center gap-4">
						<Button onClick={handlePlayPause} disabled={!isReady}>
							{isPlaying ? 'Pause' : 'Play'}
						</Button>
						<div className="text-sm text-muted-foreground">
							{isReady ? (
								<span>Ready | Offset: {offsetSeconds.toFixed(3)}s</span>
							) : (
								<span>Loading videos...</span>
							)}
						</div>
					</div>

					{/* Sync stats */}
					<div className="pt-4 border-t">
						<p className="text-xs text-muted-foreground">
							This is a prototype to validate rVFC sync works before full integration.
						</p>
					</div>
				</CardContent>
			</Card>
		</div>
	);
};

export default DualVideoTest;
