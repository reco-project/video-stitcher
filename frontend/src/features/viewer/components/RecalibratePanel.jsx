import React, { useState, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RefreshCw, Clock, AlertCircle, CheckCircle, ChevronDown } from 'lucide-react';
import { processMatchWithFrames } from '@/features/matches/api/matches';
import { useSettings } from '@/hooks/useSettings';
import { useToast } from '@/components/ui/toast';
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
	const { showToast } = useToast();
	const [timeInput, setTimeInput] = useState('');
	const [status, setStatus] = useState('idle'); // 'idle' | 'extracting' | 'processing' | 'success' | 'warning' | 'error'
	const [error, setError] = useState(null);
	const [extractorData, setExtractorData] = useState(null);
	const [isCollapsed, setIsCollapsed] = useState(true);

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
				// Use debug mode from settings
				const debugMode = settings.debugMode || false;
				const result = await processMatchWithFrames(match.id, leftBlob, rightBlob, debugMode);
				
				// Check if calibration failed but still returned default params
				if (result.calibration_failed) {
					const errorMsg = result.calibration_error || 'Not enough features found';
					setError(`Calibration failed: ${errorMsg}`);
					setStatus('warning');
					showToast({
						message: `Calibration failed: ${errorMsg}. Try a different frame with more visible features (grass, textures).`,
						type: 'warning',
						duration: 8000,
					});
				} else {
					setStatus('success');
					showToast({ message: 'Recalibration successful!', type: 'success' });
				}
				
				onRecalibrated?.(result);

				// Reset after delay
				setTimeout(() => {
					setStatus('idle');
					setError(null);
				}, 5000);
			} catch (err) {
				setError(err.message || 'Recalibration failed');
				setStatus('error');
				showToast({ message: err.message || 'Recalibration failed', type: 'error' });
			}
		},
		[match.id, onRecalibrated, settings.debugMode, showToast]
	);

	const handleExtractError = useCallback((err) => {
		setExtractorData(null);
		setError(err.message || 'Failed to extract frames');
		setStatus('error');
	}, []);

	const isProcessing = status === 'extracting' || status === 'processing';

	return (
		<>
			<div className="bg-card border rounded-lg shadow-sm overflow-hidden">
				{/* Collapsible Header */}
				<button
					onClick={() => setIsCollapsed(!isCollapsed)}
					className="w-full flex items-center justify-between px-4 py-3 bg-muted/20 hover:bg-muted/40 transition-colors"
				>
					<div className="flex items-center gap-3">
						<h4 className="font-semibold flex items-center gap-2">
							<RefreshCw className="h-4 w-4 text-muted-foreground" />
							Recalibrate
						</h4>
					</div>
					<ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`} />
				</button>

				{/* Collapsible Content */}
				{!isCollapsed && (
					<div className="px-4 py-3 border-t">
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
								Set current frame
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
						{status === 'warning' && error && (
							<div className="mt-2 flex items-start gap-1 text-xs text-yellow-600 dark:text-yellow-500">
								<AlertCircle className="h-3 w-3 mt-0.5 flex-shrink-0" />
								<span>{error} Try a different frame with more visible features (grass, textures).</span>
							</div>
						)}
						{status === 'success' && (
							<div className="mt-2 flex items-center gap-1 text-xs text-green-500">
								<CheckCircle className="h-3 w-3" />
								Recalibration complete!
							</div>
						)}
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
