import React from 'react';
import { Button } from '@/components/ui/button';
import { Play, Loader2, Eye, MoreVertical } from 'lucide-react';
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
	DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';

/**
 * MatchActionButtons centralizes conditional button logic for different match statuses
 * Uses dropdown menu to avoid button clutter
 */
export default function MatchActionButtons({
	match,
	processingId,
	deletingId,
	onProcess,
	onContinueProcess,
	onStartOver,
	onRetry,
	onReprocessAll,
	onView,
	onDelete,
	onShowError,
}) {
	const isPrimaryProcessing = processingId === match.id;
	const isDeletingThis = deletingId === match.id;

	const primary = (() => {
		// Determine primary label, click handler, and disabled state
		if (isPrimaryProcessing) return { label: 'Processing...', Icon: Loader2, onClick: null, disabled: true };

		if (match.status === 'ready' || match.status === 'warning') {
			return { label: 'View', Icon: Eye, onClick: () => onView?.(match), disabled: false };
		}

		if (match.status === 'error') {
			return {
				label: 'Retry',
				Icon: Play,
				onClick: () => onRetry?.(match.id, match.name || match.label, false),
				disabled: false,
			};
		}

		if (match.status === 'pending') {
			if (!match.src) {
				return {
					label: 'Process',
					Icon: Play,
					onClick: () => onProcess?.(match.id, match.name || match.label, false),
					disabled: false,
				};
			}
			// has video source
			if (match.processing_step === 'awaiting_frames') {
				return {
					label: 'Continue',
					Icon: Play,
					onClick: () => onContinueProcess?.(match.id, match.name || match.label),
					disabled: false,
				};
			}
			// any other pending processing sub-step: show disabled Processing
			return { label: 'Processing...', Icon: Loader2, onClick: null, disabled: true };
		}

		// fallback
		return { label: 'View', Icon: Eye, onClick: () => onView?.(match), disabled: true };
	})();

	return (
		<div className="flex gap-2 w-full">
			<Button
				onClick={(e) => {
					e.stopPropagation();
					primary.onClick && primary.onClick();
				}}
				disabled={primary.disabled}
				variant={match.status === 'ready' || match.status === 'warning' ? 'default' : 'outline'}
				size="sm"
				className="gap-1 flex-1 text-xs"
			>
				{primary.Icon && <primary.Icon className="h-3 w-3" />}
				<span>{primary.label}</span>
			</Button>

			{/* Secondary Actions Menu - Icon Only */}
			<DropdownMenu>
				<DropdownMenuTrigger asChild>
					<Button variant="outline" size="sm" className="h-9 w-9 p-0 shrink-0">
						<MoreVertical className="h-4 w-4" />
					</Button>
				</DropdownMenuTrigger>
				<DropdownMenuContent align="end" className="w-48">
					{/* Option: Continue from frames (for error status with video) */}
					{match.status === 'error' && match.src && (
						<>
							<DropdownMenuItem onClick={() => onContinueProcess?.(match.id, match.name || match.label)}>
								Continue from Frames
							</DropdownMenuItem>
							<DropdownMenuSeparator />
						</>
					)}

					{/* Option: Start Over (for pending status with video) */}
					{match.status === 'pending' && match.src && match.processing_step === 'awaiting_frames' && (
						<>
							<DropdownMenuItem onClick={() => onStartOver?.(match.id, match.name || match.label)}>
								Start Over
							</DropdownMenuItem>
							<DropdownMenuSeparator />
						</>
					)}

					{/* Option: Reprocess All (for ready or warning status) */}
					{(match.status === 'ready' || match.status === 'warning') && (
						<>
							<DropdownMenuItem
								onClick={() => onReprocessAll?.(match.id, match.name || match.label)}
								disabled={isPrimaryProcessing}
							>
								{isPrimaryProcessing ? 'Reprocessing...' : 'Reprocess All'}
							</DropdownMenuItem>
							<DropdownMenuSeparator />
						</>
					)}

					{/* Option: View Details (only for View button text variant) */}
					{match.status !== 'ready' && match.status !== 'warning' && match.status !== undefined && (
						<>
							<DropdownMenuItem
								onClick={() => onView?.(match)}
								disabled={match.status !== 'ready' && match.status !== 'warning'}
							>
								View Details
							</DropdownMenuItem>
							<DropdownMenuSeparator />
						</>
					)}

					{/* Option: View Error Details (if error exists) */}
					{match.error_message && (
						<>
							<DropdownMenuItem onClick={() => onShowError?.(match)} className="text-red-600">
								View Error Details
							</DropdownMenuItem>
							<DropdownMenuSeparator />
						</>
					)}

					{/* Option: Delete */}
					<DropdownMenuItem
						onClick={() => onDelete?.(match.id, match.name || match.label)}
						disabled={isDeletingThis}
						className="text-red-600"
					>
						{isDeletingThis ? 'Deleting...' : 'Delete'}
					</DropdownMenuItem>
				</DropdownMenuContent>
			</DropdownMenu>
		</div>
	);
}
