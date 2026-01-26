import React, { useState } from 'react';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Info, Move, Clock, Calendar, Gauge, ChevronDown } from 'lucide-react';
import { getProcessingDuration, getTranscodeMetrics, getQualitySettings } from '@/lib/matchHelpers.js';

/**
 * DescriptionPanel - Collapsible panel showing match metadata and view controls
 */
export default function DescriptionPanel({ match, yawRange, pitchRange, onYawChange, onPitchChange, saveStatus }) {
	const [isCollapsed, setIsCollapsed] = useState(false);

	// Get metrics
	const transcodeMetrics = getTranscodeMetrics(match);
	const processingDuration = getProcessingDuration(match);
	const qualitySettings = getQualitySettings(match);

	return (
		<div className="bg-card border rounded-lg shadow-sm overflow-hidden">
			{/* Collapsible Header */}
			<button
				onClick={() => setIsCollapsed(!isCollapsed)}
				className="w-full flex items-center justify-between px-4 py-3 bg-muted/20 hover:bg-muted/40 transition-colors"
			>
				<div className="flex items-center gap-3">
					<h4 className="font-semibold flex items-center gap-2">
						<Info className="h-4 w-4 text-muted-foreground" />
						Description
					</h4>
				</div>
				<div className="flex items-center gap-2">
					{/* Save Status Indicator */}
					{saveStatus && (
						<span
							className={`text-xs font-medium ${
								saveStatus === 'saving'
									? 'text-blue-600 animate-pulse'
									: saveStatus === 'success'
										? 'text-green-600'
										: 'text-red-600'
							}`}
						>
							{saveStatus === 'saving'
								? '● Saving...'
								: saveStatus === 'success'
									? '✓ Saved'
									: '✗ Save failed'}
						</span>
					)}
					<ChevronDown
						className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`}
					/>
				</div>
			</button>

			{/* Collapsible Content */}
			{!isCollapsed && (
				<>
					{/* Info Grid */}
					<div className="px-4 py-3 border-t">
						<div className="flex flex-wrap gap-x-6 gap-y-2 text-xs">
							{/* Basic Info */}
							<div className="flex items-center gap-1.5">
								<Calendar className="h-3 w-3 text-muted-foreground" />
								<span className="text-muted-foreground">Created:</span>
								<span>
									{match.created_at
										? new Date(match.created_at).toLocaleDateString('en-US', {
												month: 'short',
												day: 'numeric',
												year: 'numeric',
											})
										: 'N/A'}
								</span>
							</div>

							{/* Processing Duration */}
							{processingDuration && (
								<div className="flex items-center gap-1.5">
									<Clock className="h-3 w-3 text-muted-foreground" />
									<span className="text-muted-foreground">Processed in:</span>
									<span>{processingDuration.toFixed(1)}s</span>
								</div>
							)}

							{/* Transcode FPS */}
							{transcodeMetrics.fps && (
								<div className="flex items-center gap-1.5">
									<Gauge className="h-3 w-3 text-muted-foreground" />
									<span className="text-muted-foreground">Transcode:</span>
									<span>{transcodeMetrics.fps.toFixed(0)} fps</span>
								</div>
							)}

							{/* Audio Sync */}
							{transcodeMetrics.offsetSeconds !== undefined &&
								transcodeMetrics.offsetSeconds !== null && (
									<div className="flex items-center gap-1.5">
										<span className="text-muted-foreground">Audio Sync:</span>
										<span
											className={`font-mono ${Math.abs(transcodeMetrics.offsetSeconds) < 0.1 ? 'text-green-600' : 'text-amber-600'}`}
										>
											{transcodeMetrics.offsetSeconds > 0 ? '+' : ''}
											{transcodeMetrics.offsetSeconds.toFixed(3)}s
										</span>
									</div>
								)}

							{/* Quality Settings */}
							{qualitySettings && (
								<>
									<div className="flex items-center gap-1.5">
										<span className="text-muted-foreground">Quality:</span>
										<span className="uppercase">
											{qualitySettings.resolution || qualitySettings.preset}
										</span>
										{qualitySettings.bitrate && (
											<span className="text-muted-foreground">@ {qualitySettings.bitrate}</span>
										)}
									</div>
									<div className="flex items-center gap-1.5">
										<span className="text-muted-foreground">GPU Decoding:</span>
										<span className={qualitySettings.use_gpu_decode ? 'text-green-600' : ''}>
											{qualitySettings.use_gpu_decode ? 'On' : 'Off'}
										</span>
									</div>
								</>
							)}

							{/* Lens Profiles */}
							{(match.left_videos?.[0]?.profile_id || match.metadata?.left_profile_id) && (
								<div className="flex items-center gap-1.5 basis-full mt-1">
									<span className="text-muted-foreground">Profiles:</span>
									<code className="px-1.5 py-0.5 bg-muted rounded text-[10px]">
										{match.left_videos?.[0]?.profile_id || match.metadata?.left_profile_id}
									</code>
									<span className="text-muted-foreground">/</span>
									<code className="px-1.5 py-0.5 bg-muted rounded text-[10px]">
										{match.right_videos?.[0]?.profile_id || match.metadata?.right_profile_id}
									</code>
								</div>
							)}
						</div>
					</div>

					{/* Controls Section */}
					<div className="px-4 py-3 border-t bg-muted/10">
						<div className="flex flex-wrap items-center gap-6">
							{/* View Range Controls */}
							<div className="flex items-center gap-4">
								<Move className="h-4 w-4 text-muted-foreground" />
								<div className="flex items-center gap-2">
									<Label className="text-xs text-muted-foreground">H:</Label>
									<Slider
										min={30}
										max={180}
										step={5}
										value={[yawRange]}
										onValueChange={(value) => onYawChange(value[0])}
										className="w-24"
									/>
									<span className="text-xs font-mono w-8">{yawRange}°</span>
								</div>
								<div className="flex items-center gap-2">
									<Label className="text-xs text-muted-foreground">V:</Label>
									<Slider
										min={5}
										max={60}
										step={5}
										value={[pitchRange]}
										onValueChange={(value) => onPitchChange(value[0])}
										className="w-24"
									/>
									<span className="text-xs font-mono w-8">{pitchRange}°</span>
								</div>
							</div>
						</div>
					</div>
				</>
			)}
		</div>
	);
}
