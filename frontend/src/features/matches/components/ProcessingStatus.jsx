import React from 'react';
import { useState, useEffect } from 'react';
import { Loader2, CheckCircle, XCircle, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

/**
 * Component to display match processing status with better UX
 */
export default function ProcessingStatus({ status }) {
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
		if (typeof status.progress_percent === 'number') return Math.round(status.progress_percent);
		if (typeof status.processing_percent === 'number') return Math.round(status.processing_percent);
		if (typeof status.step_progress === 'number') return Math.round(status.step_progress);

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
					title: getStepTitle(status.processing_step),
					message: status.processing_message || 'Synchronizing audio and stacking videos...',
					animated: true,
					bgClass: 'bg-blue-50 dark:bg-blue-950',
				};
			case 'calibrating':
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: getStepTitle(status.processing_step),
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
				return null;
		}
	};

	const getStepTitle = (step) => {
		switch (step) {
			case 'initializing':
				return 'Initializing';
			case 'transcoding':
				return 'Syncing Videos';
			case 'extracting_frame':
				return 'Extracting Frame';
			case 'feature_matching':
				return 'Matching Features';
			case 'optimizing':
				return 'Optimizing Calibration';
			case 'complete':
				return 'Complete';
			default:
				return 'Processing';
		}
	};

	const getCalibrationMessage = (step) => {
		switch (step) {
			case 'feature_matching':
				return 'Detecting and matching features...';
			case 'optimizing':
			case 'position_optimization':
				return 'Optimizing camera positions...';
			default:
				return 'Calibrating cameras...';
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

					{/* Progress bar for active processing */}
					{isProcessing && (
						<div className="mb-3">
							<div className="flex justify-between items-center mb-1">
								<span className="text-xs font-medium opacity-70">Overall Progress</span>
								<span className="text-xs opacity-70">{progress}%</span>
							</div>
							<Progress value={progress} className="h-2" />
						</div>
					)}

					{/* Detailed metrics: fps, frames, audio sync, elapsed/remaining */}
					<div className="text-xs opacity-80 space-y-1 mt-2">
						{fps && (
							<div>
								Transcoding FPS: <span className="font-medium">{fps}</span>
							</div>
						)}
						{framesInfo.processed != null && framesInfo.total != null && (
							<div>
								Frames: <span className="font-medium">{framesInfo.processed}</span> /{' '}
								<span className="font-medium">{framesInfo.total}</span>
							</div>
						)}
						{status.audio_sync && (
							<div>
								Audio Sync:{' '}
								<span className="font-medium">{status.audio_sync.status || status.audio_sync}</span>
								{status.audio_sync.progress != null
									? ` — ${Math.round(status.audio_sync.progress)}%`
									: ''}
							</div>
						)}
						{elapsed != null && (
							<div>
								Elapsed: <span className="font-medium">{formatDuration(elapsed)}</span>
								{remaining != null && (
									<>
										{' '}
										— Remaining: <span className="font-medium">{formatDuration(remaining)}</span>
									</>
								)}
							</div>
						)}
					</div>

					{/* Timestamps */}
					<div className="space-y-1">
						{status.processing_started_at && status.status !== 'pending' && (
							<div className="text-xs opacity-60">
								Started: {new Date(status.processing_started_at).toLocaleString()}
							</div>
						)}
						{status.processing_completed_at && (
							<div className="text-xs opacity-60">
								Completed: {new Date(status.processing_completed_at).toLocaleString()}
							</div>
						)}
					</div>

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
										Hide Details
									</>
								) : (
									<>
										<ChevronDown className="h-3 w-3" />
										Show Error Details
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
