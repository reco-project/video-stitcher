import React from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, CheckCircle, XCircle, Clock, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '@/components/ui/button';

/**
 * Component to display match processing status
 */
export default function ProcessingStatus({ status }) {
	const [showDetails, setShowDetails] = React.useState(false);

	if (!status) return null;

	const getStatusConfig = () => {
		switch (status.status) {
			case 'pending':
				return {
					icon: Clock,
					variant: 'default',
					color: 'text-gray-500',
					title: 'Pending',
					message: 'Match is ready to be processed',
				};
			case 'transcoding':
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: getStepTitle(status.processing_step),
					message: status.processing_message || 'Synchronizing audio and stacking videos...',
					animated: true,
				};
			case 'calibrating':
				return {
					icon: Loader2,
					variant: 'default',
					color: 'text-blue-500',
					title: getStepTitle(status.processing_step),
					message: status.processing_message || getCalibrationMessage(status.processing_step),
					animated: true,
				};
			case 'ready':
				return {
					icon: CheckCircle,
					variant: 'success',
					color: 'text-green-500',
					title: 'Ready',
					message: 'Match is ready to view',
				};
			case 'error':
				return {
					icon: XCircle,
					variant: 'destructive',
					color: 'text-red-500',
					title: 'Processing Failed',
					message: status.error_message || 'An unknown error occurred',
					errorCode: status.error_code,
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

	return (
		<Alert variant={config.variant} className="mb-4">
			<div className="flex items-start gap-3">
				<Icon className={`h-5 w-5 ${config.color} ${config.animated ? 'animate-spin' : ''} shrink-0 mt-0.5`} />
				<div className="flex-1">
					<div className="font-semibold mb-1">{config.title}</div>
					<AlertDescription>
						{config.message}
						{config.errorCode && (
							<div className="text-xs mt-1 opacity-70">Error code: {config.errorCode}</div>
						)}
					</AlertDescription>
					{status.processing_started_at && status.status !== 'pending' && (
						<div className="text-xs mt-2 opacity-60">
							Started: {new Date(status.processing_started_at).toLocaleString()}
						</div>
					)}
					{status.processing_completed_at && (
						<div className="text-xs opacity-60">
							Completed: {new Date(status.processing_completed_at).toLocaleString()}
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
		</Alert>
	);
}
