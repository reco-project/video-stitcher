import React from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

/**
 * ErrorDetailsDialog shows detailed error information for failed matches
 * Separated from MatchList for better component separation
 */
export default function ErrorDetailsDialog({ open, onOpenChange, match }) {
	if (!match) return null;

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="max-w-3xl">
				<DialogHeader>
					<DialogTitle>Error Details: {match.name || match.label}</DialogTitle>
					<DialogDescription>Complete error information for this match</DialogDescription>
				</DialogHeader>

				<div className="space-y-4">
					{match.error_code && (
						<div>
							<div className="text-sm font-medium mb-1">Error Code</div>
							<div className="p-2 bg-muted rounded text-sm font-mono">{match.error_code}</div>
						</div>
					)}

					<div>
						<div className="text-sm font-medium mb-1">Error Message</div>
						<div className="p-3 bg-muted rounded text-sm font-mono overflow-x-auto max-h-96 overflow-y-auto">
							<pre className="whitespace-pre-wrap wrap-break-word">
								{match.error_message || 'No error message available'}
							</pre>
						</div>
					</div>

					{match.processing_started_at && (
						<div className="text-xs text-muted-foreground">
							Processing started: {new Date(match.processing_started_at).toLocaleString()}
						</div>
					)}
					{match.processing_completed_at && (
						<div className="text-xs text-muted-foreground">
							Failed at: {new Date(match.processing_completed_at).toLocaleString()}
						</div>
					)}
				</div>

				<div className="flex justify-end">
					<Button onClick={() => onOpenChange(false)} variant="outline">
						Close
					</Button>
				</div>
			</DialogContent>
		</Dialog>
	);
}
