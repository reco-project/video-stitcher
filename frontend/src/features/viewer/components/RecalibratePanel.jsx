import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RefreshCw, Clock, AlertCircle, CheckCircle } from 'lucide-react';
import { processMatchWithFrames } from '@/features/matches/api/matches';
import { useSettings } from '@/hooks/useSettings';
import FrameExtractor from './FrameExtractor';

/**
 * Parse time string in format MM:SS or HH:MM:SS to seconds
 */
function parseTimeToSeconds(timeStr) {
	const parts = timeStr.split(':').map((p) => parseInt(p, 10));
	if (parts.some(isNaN)) return null;

	if (parts.length === 2) {
		// MM:SS
		return parts[0] * 60 + parts[1];
	} else if (parts.length === 3) {
		// HH:MM:SS
		return parts[0] * 3600 + parts[1] * 60 + parts[2];
	}
	return null;
}

/**
 * Format seconds to MM:SS or HH:MM:SS
 */
function formatSecondsToTime(seconds) {
	const h = Math.floor(seconds / 3600);
	const m = Math.floor((seconds % 3600) / 60);
	const s = Math.floor(seconds % 60);

	if (h > 0) {
		return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
	}
	return `${m}:${String(s).padStart(2, '0')}`;
}

/**
 * RecalibratePanel - Allows user to recalibrate at a specific time
 */
export default function RecalibratePanel({ match, videoRef, onRecalibrated }) {
	const { settings } = useSettings();
	const [timeInput, setTimeInput] = useState('');
	const [status, setStatus] = useState('idle'); // 'idle' | 'extracting' | 'processing' | 'success' | 'error'
	const [error, setError] = useState(null);
	const [extractorData, setExtractorData] = useState(null);

    // TODO: little ugly - move video URL construction to a util
	// Build full video URL from match.src
	const getVideoUrl = useCallback(() => {
		if (!match?.src) return null;
		const apiBaseUrl = settings.apiBaseUrl || 'http://127.0.0.1:8000/api';
		const baseUrl = apiBaseUrl.replace('/api', '');
		return `${baseUrl}/${match.src}`;
	}, [match?.src, settings.apiBaseUrl]);

	// Default to 10% of video duration (avoids shaky start)
	useEffect(() => {
		if (videoRef?.duration && !timeInput) {
			const defaultTime = Math.floor(videoRef.duration * 0.1);
			setTimeInput(formatSecondsToTime(defaultTime));
		}
	}, [videoRef?.duration]);

	const handleUseCurrentTime = () => {
		if (videoRef?.currentTime) {
			setTimeInput(formatSecondsToTime(videoRef.currentTime));
		}
	};

	const handleRecalibrate = useCallback(() => {
		const seconds = parseTimeToSeconds(timeInput);
		if (seconds === null) {
			setError('Invalid time format. Use MM:SS or HH:MM:SS');
			setStatus('error');
			return;
		}

		if (videoRef && seconds > videoRef.duration) {
			setError(`Time exceeds video duration (${formatSecondsToTime(videoRef.duration)})`);
			setStatus('error');
			return;
		}

		const videoUrl = getVideoUrl();
		if (!videoUrl) {
			setError('Video source not available');
			setStatus('error');
			return;
		}

		setError(null);
		setStatus('extracting');
		setExtractorData({
			frameTime: seconds,
			leftUniforms: match.left_uniforms,
			rightUniforms: match.right_uniforms,
			videoSrc: videoUrl,
		});
	}, [timeInput, videoRef, match, getVideoUrl]);

	const handleExtractComplete = useCallback(
		async ({ leftBlob, rightBlob }) => {
			setExtractorData(null);
			setStatus('processing');

			try {
				const result = await processMatchWithFrames(match.id, leftBlob, rightBlob, false);
				setStatus('success');
				onRecalibrated?.(result);

				// Reset after success
				setTimeout(() => {
					setStatus('idle');
				}, 3000);
			} catch (err) {
				setError(err.message || 'Recalibration failed');
				setStatus('error');
			}
		},
		[match.id, onRecalibrated]
	);

	const handleExtractError = useCallback((err) => {
		setExtractorData(null);
		setError(err.message || 'Failed to extract frames');
		setStatus('error');
	}, []);

	const isProcessing = status === 'extracting' || status === 'processing';

	return (
		<>
			<div className="border-t pt-3">
				<h4 className="text-xs font-semibold mb-2 flex items-center gap-2">
					<RefreshCw className="h-3 w-3" />
					Recalibrate
				</h4>
				<p className="text-xs text-muted-foreground mb-3">
					Re-run calibration using a frame at a specific time in the video.
				</p>

				<div className="flex items-end gap-2">
					<div className="flex-1">
						<Label htmlFor="recalibrate-time" className="text-xs">
							Time (MM:SS)
						</Label>
						<Input
							id="recalibrate-time"
							type="text"
							placeholder="0:30"
							value={timeInput}
							onChange={(e) => {
								setTimeInput(e.target.value);
								if (status === 'error') setStatus('idle');
							}}
							disabled={isProcessing}
							className="h-8 text-xs"
						/>
					</div>
					<Button
						type="button"
						variant="outline"
						size="sm"
						onClick={handleUseCurrentTime}
						disabled={isProcessing || !videoRef?.currentTime}
						className="h-8"
						title="Use current video time"
					>
						<Clock className="h-3 w-3" />
					</Button>
					<Button
						type="button"
						size="sm"
						onClick={handleRecalibrate}
						disabled={isProcessing || !timeInput}
						className="h-8"
					>
						{isProcessing ? (
							<>
								<RefreshCw className="h-3 w-3 mr-1 animate-spin" />
								{status === 'extracting' ? 'Extracting...' : 'Calibrating...'}
							</>
						) : (
							'Recalibrate'
						)}
					</Button>
				</div>

				{/* Status messages */}
				{status === 'error' && error && (
					<div className="mt-2 flex items-center gap-1 text-xs text-red-500">
						<AlertCircle className="h-3 w-3" />
						{error}
					</div>
				)}
				{status === 'success' && (
					<div className="mt-2 flex items-center gap-1 text-xs text-green-500">
						<CheckCircle className="h-3 w-3" />
						Recalibration complete! Refresh to see changes.
					</div>
				)}
			</div>

			{/* Frame Extractor Modal */}
			{extractorData && (
				<FrameExtractor
					videoSrc={extractorData.videoSrc}
					frameTime={extractorData.frameTime}
					leftUniforms={extractorData.leftUniforms}
					rightUniforms={extractorData.rightUniforms}
					onComplete={handleExtractComplete}
					onError={handleExtractError}
				/>
			)}
		</>
	);
}
