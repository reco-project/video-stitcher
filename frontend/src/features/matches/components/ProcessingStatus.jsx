import React, { useState, useEffect } from 'react';
import { Loader2, CheckCircle, XCircle, Clock, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';

/**
 * Component to display match processing status with better UX
 */
export default function ProcessingStatus({ status, onComplete }) {
	const [showDetails, setShowDetails] = React.useState(false);
	const [now, setNow] = useState(Date.now());

	// Keep a ticking clock for elapsed/remaining estimates
	useEffect(() => {
		const t = setInterval(() => setNow(Date.now()), 1000);
		return () => clearInterval(t);
	}, []);

	if (!status) return null;

	// Determine progress percentage based on step or explicit fields returned from backend
	const getProgress = () => {
		// Use transcoding progress if available
		if (status.status === 'transcoding' && typeof status.transcode_progress === 'number') {
			return Math.round(status.transcode_progress);
		}

		if (typeof status.progress_percent === 'number') return Math.round(status.progress_percent);
		if (typeof status.processing_percent === 'number') return Math.round(status.processing_percent);
		if (typeof status.step_progress === 'number') return Math.round(status.step_progress);

		// For transcoding status without specific progress, return 0 to show progress bar
		if (status.status === 'transcoding') return 0;

		// Calibrating is near the end of processing
		if (status.status === 'calibrating') return 95;

		const steps = ['initializing', 'transcoding', 'extracting_frame', 'feature_matching', 'optimizing', 'complete'];
		const stepIndex = steps.indexOf(status.processing_step || '');
		if (stepIndex === -1) return 0;
		return Math.round((stepIndex / steps.length) * 100);
	};

	const getFps = () => {
		if (typeof status.fps === 'number') return status.fps;
		if (status.metrics && typeof status.metrics.fps === 'number') return status.metrics.fps;
		return null;
	};

	const getFramesInfo = () => {
		const processed = status.frames_processed || (status.metrics && status.metrics.frames_processed) || null;
		const total = status.frames_total || (status.metrics && status.metrics.frames_total) || null;
		return { processed, total };
	};

	const formatDuration = (seconds) => {
		if (seconds === null || isNaN(seconds)) return '--:--';
		const s = Math.max(0, Math.round(seconds));
		const hh = Math.floor(s / 3600);
		const mm = Math.floor((s % 3600) / 60);
		const ss = s % 60;
		if (hh > 0) return `${hh}h ${mm}m ${ss}s`;
		if (mm > 0) return `${mm}m ${ss}s`;
		return `${ss}s`;
	};

	const getElapsedSeconds = () => {
		if (!status.processing_started_at) return null;
		const started = new Date(status.processing_started_at).getTime();
		return Math.max(0, (now - started) / 1000);
	};

	const estimateRemainingSeconds = (elapsed, percent, fps, framesInfo) => {
		// For transcoding, use time-based estimate if available
		if (status.status === 'transcoding' && status.transcode_current_time && status.transcode_total_duration) {
			const remaining = status.transcode_total_duration - status.transcode_current_time;
			// Adjust by speed if available
			if (status.transcode_speed) {
				const speed = parseFloat(status.transcode_speed);
				if (speed > 0) {
					return remaining / speed;
				}
			}
			return remaining;
		}

		// Prefer frame-based estimate when fps & frames info available
		if (
			fps &&
			framesInfo.processed != null &&
			framesInfo.total != null &&
			framesInfo.total > framesInfo.processed
		) {
			const remainingFrames = framesInfo.total - framesInfo.processed;
			return remainingFrames / fps;
		}
		// Fallback to percent-based estimate
		if (percent > 0 && percent < 100 && elapsed != null) {
			return Math.max(0, elapsed * (100 / percent) - elapsed);
		}
		return null;
	};

	const getStatusConfig = () => {
		switch (status.status) {
			case 'pending':
				return {
					icon: Clock,
					variant: 'default',
					color: 'text-gray-500',
					title: 'Pending',
					message: 'Match is ready to be processed',
					bgClass: 'bg-gray-50 dark:bg-gray-950',
				};
			case 'transcoding':
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: 'Syncing Videos',
					message: status.processing_message || 'Synchronizing audio and encoding side-by-side video...',
					animated: true,
					bgClass: 'bg-blue-50 dark:bg-blue-950',
				};
			case 'calibrating':
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: 'Calibrating Cameras',
					message: status.processing_message || getCalibrationMessage(status.processing_step),
					animated: true,
					bgClass: 'bg-blue-50 dark:bg-blue-950',
				};
			case 'ready':
				return {
					icon: CheckCircle,
					variant: 'success',
					color: 'text-green-500',
					title: 'Ready',
					message: 'Match is ready to view',
					bgClass: 'bg-green-50 dark:bg-green-950',
				};
			case 'error':
				return {
					icon: XCircle,
					variant: 'destructive',
					color: 'text-red-500',
					title: 'Processing Failed',
					message: status.error_message || 'An unknown error occurred',
					errorCode: status.error_code,
					bgClass: 'bg-red-50 dark:bg-red-950',
				};
			default:
				// Fallback for any unknown status - treat as processing
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: 'Processing',
					message: status.processing_message || status.message || 'Processing...',
					animated: true,
					bgClass: 'bg-blue-50 dark:bg-blue-950',
				};
		}
	};

	const getCalibrationMessage = (step) => {
		switch (step) {
			case 'feature_matching':
				return 'Analyzing camera features and finding matches...';
			case 'optimizing':
				return 'Computing optimal camera positions and alignment...';
			default:
				return 'Running camera calibration...';
		}
	};

	const config = getStatusConfig();
	if (!config) return null;

	const Icon = config.icon;
	const progress = getProgress();
	const isProcessing =
		status.status === 'transcoding' || status.status === 'calibrating' || status.status === 'processing';
	const fps = getFps();
	const framesInfo = getFramesInfo();
	const elapsed = getElapsedSeconds();
	const remaining = estimateRemainingSeconds(elapsed, progress, fps, framesInfo);

	return (
		<div className={`rounded-lg border p-4 ${config.bgClass} transition-colors`}>
			<div className="flex items-start gap-3">
				<Icon className={`h-5 w-5 ${config.color} ${config.animated ? 'animate-spin' : ''} shrink-0 mt-0.5`} />
				<div className="flex-1">
					<div className="font-semibold mb-1">{config.title}</div>
					<div className="text-sm opacity-90 mb-3">
						{config.message}
						{config.errorCode && (
							<div className="text-xs mt-1 opacity-70">Error code: {config.errorCode}</div>
						)}
					</div>

					{/* Progress bar for active processing - show immediately after message */}
					{isProcessing && (
						<div className="mb-3">
							<div className="flex justify-between items-center mb-1">
								<span className="text-xs font-medium opacity-70">Progress</span>
								<span className="text-xs opacity-70">{progress}%</span>
							</div>
							<Progress value={progress || 0} className="h-2" />
						</div>
					)}

					{/* Show encoder info and transcoding metrics when available */}
					{status.transcode_fps != null && status.transcode_fps > 0 && (
						<div className="text-xs opacity-80 mb-3 space-y-1">
							{/* Encoder badge (info only, no action during transcoding) */}
							{status.transcode_encoder && (
								<div className="flex items-center gap-2 mb-2">
									<Badge
										variant={status.transcode_encoder === 'libx264' ? 'secondary' : 'default'}
										className="gap-1"
										title={
											status.transcode_encoder === 'libx264'
												? 'Software encoding'
												: 'Hardware-accelerated encoding'
										}
									>
										<Zap className="h-3 w-3" />
										{status.transcode_encoder === 'libx264'
											? 'CPU Encoder'
											: status.transcode_encoder === 'h264_nvenc'
												? 'NVIDIA GPU'
												: status.transcode_encoder === 'h264_qsv'
													? 'Intel GPU'
													: status.transcode_encoder === 'h264_amf'
														? 'AMD GPU'
														: status.transcode_encoder}
									</Badge>
								</div>
							)}
							<div className="flex items-center gap-4">
								<div>
									<span className="opacity-70">Encoding: </span>
									<span className="font-mono font-semibold">{status.transcode_fps} FPS</span>
								</div>
								{status.transcode_speed && (
									<div>
										<span className="opacity-70">•</span>
										<span className="font-mono font-semibold ml-1">
											{status.transcode_speed}x speed
										</span>
									</div>
								)}
								{status.status === 'transcoding' &&
									status.transcode_current_time &&
									status.transcode_total_duration && (
										<div>
											<span className="opacity-70">•</span>
											<span className="font-mono font-semibold ml-1">
												{(() => {
													const remaining =
														status.transcode_total_duration - status.transcode_current_time;
													const speed = parseFloat(status.transcode_speed || 1);
													const estimatedSec = remaining / speed;
													const mins = Math.floor(estimatedSec / 60);
													const secs = Math.floor(estimatedSec % 60);
													return `~${mins}:${secs.toString().padStart(2, '0')} left`;
												})()}
											</span>
										</div>
									)}
							</div>
							{/* Show video position when available */}
							{status.transcode_current_time != null &&
								status.transcode_total_duration != null &&
								status.transcode_total_duration > 0 && (
									<div className="opacity-70">
										Video position: {formatDuration(status.transcode_current_time)} /{' '}
										{formatDuration(status.transcode_total_duration)}
									</div>
								)}
						</div>
					)}

					{/* Completion button for ready status */}
					{status.status === 'ready' && onComplete && (
						<div className="mt-3">
							<Button onClick={onComplete} size="lg" className="w-full">
								View Match
							</Button>
						</div>
					)}

					{/* Show elapsed time for processing */}
					{isProcessing && elapsed != null && (
						<div className="text-xs opacity-70 mb-2">
							Elapsed: {formatDuration(elapsed)}
							{remaining != null && <> • Est. remaining: {formatDuration(remaining)}</>}
						</div>
					)}

					{/* Show detailed error logs for errors */}
					{status.status === 'error' && status.error_message && (
						<div className="mt-3">
							<Button
								variant="outline"
								size="sm"
								onClick={() => setShowDetails(!showDetails)}
								className="gap-1 h-7 text-xs"
							>
								{showDetails ? (
									<>
										<ChevronUp className="h-3 w-3" />
										Hide Error
									</>
								) : (
									<>
										<ChevronDown className="h-3 w-3" />
										Show Error
									</>
								)}
							</Button>

							{showDetails && (
								<div className="mt-2 p-3 bg-muted/50 rounded border text-xs font-mono overflow-x-auto max-h-64 overflow-y-auto">
									<pre className="whitespace-pre-wrap break-words">{status.error_message}</pre>
								</div>
							)}
						</div>
					)}
				</div>
			</div>
		</div>
	);
}
