import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { X, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/cn';

/**
 * StatusBar shows persistent background operations and processing feedback
 * Non-blocking, always visible at bottom of application
 */
export default function StatusBar() {
	const [operations, setOperations] = useState([]);

	// In Phase 2, this will be connected to a global state management (Zustand/Context)
	// For now, it's a placeholder that can be populated from parent components

	useEffect(() => {
		// Subscribe to operation updates (this will be wired up in Phase 2)
		// If an operation with the same id already exists, update it instead of appending.
		const handleAddOperation = (event) => {
			if (!event.detail) return;
			const incoming = event.detail;

			setOperations((prev) => {
				const exists = prev.some((op) => op.id === incoming.id);
				if (exists) {
					return prev.map((op) => (op.id === incoming.id ? { ...op, ...incoming } : op));
				}
				return [...prev, incoming];
			});

			// If operation is complete or errored, schedule auto-removal
			if (incoming.status === 'success' || incoming.status === 'error') {
				const removeAfter = typeof incoming.autoDismiss === 'number' ? incoming.autoDismiss : 3500;
				setTimeout(() => {
					setOperations((prev) => prev.filter((op) => op.id !== incoming.id));
				}, removeAfter);
			}
		};

		window.addEventListener('addOperation', handleAddOperation);
		return () => window.removeEventListener('addOperation', handleAddOperation);
	}, []);

	const removeOperation = (id) => {
		setOperations((prev) => prev.filter((op) => op.id !== id));
	};

	const getStatusIcon = (status) => {
		switch (status) {
			case 'loading':
			case 'processing':
				return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
			case 'success':
				return <CheckCircle className="h-4 w-4 text-green-500" />;
			case 'error':
				return <AlertCircle className="h-4 w-4 text-red-500" />;
			default:
				return null;
		}
	};

	if (operations.length === 0) return null;

	return (
		// Fixed container so the status appears above footer/side controls and stays visible
		<div className="fixed left-1/2 transform -translate-x-1/2 bottom-16 w-full max-w-3xl z-50 pointer-events-auto">
			<div className="border bg-card/60 backdrop-blur-sm rounded-lg shadow-lg px-4 py-3 space-y-2 max-h-56 overflow-y-auto">
				{operations.map((op) => (
					<div
						key={op.id}
						className={cn(
							'flex items-center justify-between gap-3 p-3 rounded-md',
							op.status === 'success' && 'bg-green-50 dark:bg-green-950',
							op.status === 'error' && 'bg-red-50 dark:bg-red-950',
							(op.status === 'loading' || op.status === 'processing') && 'bg-blue-50 dark:bg-blue-950'
						)}
					>
						<div className="flex items-center gap-3 flex-1">
							{getStatusIcon(op.status)}
							<div className="flex-1">
								<p className="text-sm font-medium">{op.message}</p>
								{op.details && <p className="text-xs text-muted-foreground mt-1">{op.details}</p>}
								{op.progress !== undefined && (
									<div className="w-full bg-muted rounded h-1.5 mt-2 overflow-hidden">
										<div
											className="bg-blue-600 h-full transition-all"
											style={{ width: `${op.progress}%` }}
										/>
									</div>
								)}
							</div>
						</div>

						{op.status === 'success' || op.status === 'error' ? (
							<Button
								variant="ghost"
								size="sm"
								onClick={() => removeOperation(op.id)}
								className="h-6 w-6 p-0"
							>
								<X className="h-4 w-4" />
							</Button>
						) : null}
					</div>
				))}
			</div>
		</div>
	);
}

/**
 * Helper function to dispatch operations to StatusBar
 * Usage: notifyStatusBar({ message: 'Processing...', status: 'loading' })
 */
export function notifyStatusBar(operation) {
	const id = operation.id || `op-${Date.now()}`;
	const event = new CustomEvent('addOperation', {
		detail: { id, ...operation },
	});
	window.dispatchEvent(event);
	return id;
}
