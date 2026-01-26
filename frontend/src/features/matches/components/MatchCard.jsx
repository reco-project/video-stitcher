import React, { useEffect, useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Clock, CheckCircle, XCircle, AlertTriangle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/cn';

/**
 * MatchCard displays a single match with status, metadata, and preview
 * Designed for grid layout (responsive: 3-col desktop, 2-col tablet, 1-col mobile)
 */
export default function MatchCard({ match, onSelect, className }) {
	const [previewUrl, setPreviewUrl] = useState(null);
	const [previewLoading, setPreviewLoading] = useState(true);

	// Use viewed state from backend match model (already includes localStorage sync from Viewer)
	const viewed = match.viewed || false;

	// Try to load preview image after transcoding
	useEffect(() => {
		if (match.src && (match.status === 'ready' || match.status === 'warning')) {
			// Video is ready, preview should exist
			const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api';
			const baseUrl = apiBaseUrl.replace('/api', '');

			const previewUrl = `${baseUrl}/${match.src.replace(/\.[^.]+$/, '_preview.jpg')}`;

			const img = new Image();
			img.onload = () => {
				setPreviewUrl(previewUrl);
				setPreviewLoading(false);
			};
			img.onerror = () => {
				setPreviewLoading(false);
				setPreviewUrl(null);
			};
			img.src = previewUrl;
		} else {
			setPreviewLoading(false);
			setPreviewUrl(null);
		}
	}, [match.src, match.status, match.id]);

	const getStatusBadge = (status, viewed) => {
		// Check if match is actually complete (has all required data)
		const hasRequiredData = match.src && match.params && match.left_uniforms && match.right_uniforms;
		// Match is ready if status says so, OR if it has all data AND isn't awaiting frames
		const isActuallyReady = status === 'ready' || status === 'warning' || (hasRequiredData && match.processing_step !== 'awaiting_frames');

		// Show warning badge for calibration failure
		if (status === 'warning') {
			return (
				<Badge variant="outline" className="gap-1 border-yellow-500 text-yellow-600 dark:text-yellow-400">
					<AlertTriangle className="h-3 w-3" />
					Calibration Failed
				</Badge>
			);
		}

		// If it has all data and isn't waiting for frames, treat as ready
		if (isActuallyReady) {
			return !viewed ? (
				<Badge variant="default" className="gap-1 bg-green-500">
					<CheckCircle className="h-3 w-3" />
					Ready
				</Badge>
			) : null;
		}

		switch (status) {
			case 'pending':
				return (
					<Badge variant="secondary" className="gap-1">
						<Clock className="h-3 w-3" />
						Pending
					</Badge>
				);
			case 'transcoding':
				return (
					<Badge variant="default" className="gap-1 bg-blue-500">
						<Loader2 className="h-3 w-3 animate-spin" />
						Syncing
					</Badge>
				);
			case 'calibrating':
				return (
					<Badge variant="default" className="gap-1 bg-blue-500">
						<Loader2 className="h-3 w-3 animate-spin" />
						Calibrating
					</Badge>
				);
			case 'error':
				return (
					<Badge variant="destructive" className="gap-1">
						<XCircle className="h-3 w-3" />
						Error
					</Badge>
				);
			default:
				return null;
		}
	};

	return (
		<div onClick={onSelect} className={cn('flex flex-col gap-3 cursor-pointer h-full', className)}>
			{/* Preview Thumbnail - Full width for grid with consistent aspect ratio */}
			<div className="w-full aspect-video rounded-md bg-muted overflow-hidden border flex-shrink-0">
				{previewLoading ? (
					<div className="w-full h-full flex items-center justify-center">
						<Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
					</div>
				) : previewUrl ? (
					<img
						src={previewUrl}
						alt={match.name}
						className="w-full h-full object-cover"
						onError={() => setPreviewUrl(null)}
					/>
				) : (
					<div className="w-full h-full flex items-center justify-center text-muted-foreground text-4xl">
						🎬
					</div>
				)}
			</div>

			{/* Match Info */}
			<div className="flex-1">
				<div className="flex items-start justify-between gap-2 mb-2">
					<h3 className="font-semibold line-clamp-2 text-sm">{match.name || match.label}</h3>
					{match.status && getStatusBadge(match.status, viewed)}
				</div>

				<div className="text-xs text-muted-foreground space-y-1">
					{match.created_at && <p>📅 {new Date(match.created_at).toLocaleDateString()}</p>}
					{match.src && <p className="line-clamp-1">📁 {match.src.split('/').pop()}</p>}
					{match.processing_time_ms && <p>⏱️ {(match.processing_time_ms / 1000).toFixed(1)}s</p>}

					{/* Show transcoding metrics when available */}
					{(match.status === 'transcoding' || match.processing_step === 'transcoding') && (
						<>
							{typeof match.fps === 'number' && <p>⚡ FPS: {match.fps}</p>}
							{match.frames_processed != null && match.frames_total != null ? (
								<p>
									🎞 Frames: {match.frames_processed} / {match.frames_total} (
									{Math.round((match.frames_processed / Math.max(1, match.frames_total)) * 100)}%)
								</p>
							) : match.progress_percent != null ? (
								<p>Progress: {Math.round(match.progress_percent)}%</p>
							) : null}
						</>
					)}
					{match.status === 'error' && match.error_message && (
						<p className="text-red-500/70 line-clamp-1">❌ {match.error_message}</p>
					)}
				</div>
			</div>
		</div>
	);
}
