import React from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import ProcessingStatus from './ProcessingStatus';
import { Loader2, X } from 'lucide-react';

/**
 * ProcessingDialog shows detailed status of match processing
 * Always stay open during processing, closeable only after completion
 */
export default function ProcessingDialog({ open, onOpenChange, matchName, processingStatus }) {
	// Can only close if processing is done or errored
	const canClose =
		processingStatus?.status === 'ready' ||
		processingStatus?.status === 'warning' ||
		processingStatus?.status === 'error';

	// Don't allow clicking outside to close during processing
	const handleOpenChange = (newOpen) => {
		if (newOpen === false && !canClose) {
			return; // Prevent closing
		}
		onOpenChange(newOpen);
	};

	return (
		<Dialog open={open} onOpenChange={handleOpenChange}>
			<DialogContent className="max-w-2xl" onEscapeKeyDown={(e) => !canClose && e.preventDefault()}>
				<DialogHeader>
					<div className="flex items-center justify-between">
						<div>
							<DialogTitle>Processing Match: {matchName}</DialogTitle>
							<DialogDescription className="mt-1.5">
								Transcoding videos and calibrating camera parameters for stitching
							</DialogDescription>
						</div>
						{canClose && (
							<Button
								variant="ghost"
								size="sm"
								className="h-8 w-8 p-0"
								onClick={() => onOpenChange(false)}
							>
								<X className="h-4 w-4" />
							</Button>
						)}
					</div>
				</DialogHeader>

				<div className="py-4 min-h-24">
					{processingStatus ? (
						<ProcessingStatus status={processingStatus} />
					) : (
						<div className="flex items-center justify-center p-8">
							<Loader2 className="h-8 w-8 animate-spin text-primary" />
							<span className="ml-2">Starting processing...</span>
						</div>
					)}
				</div>

				<div className="flex justify-end gap-2 pt-4 border-t">
					{canClose ? (
						<Button onClick={() => onOpenChange(false)} variant="default">
							Done
						</Button>
					) : (
						<p className="text-xs text-muted-foreground">
							Keep this window open while processing. It&apos;s safe to use the app in the background.
						</p>
					)}
				</div>
			</DialogContent>
		</Dialog>
	);
}
