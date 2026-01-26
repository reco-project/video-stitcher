import React, { useState, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Palette, RotateCcw, Wand2, ChevronDown, Check, Settings2, Blend } from 'lucide-react';
import { DEFAULT_COLOR_CORRECTION } from '../utils/utils';
import { autoColorCorrection } from '@/features/matches/api/matches.js';
import { useViewerStore } from '../stores/store.js';

/**
 * Check if LAB correction is applied (not identity)
 */
function hasLabCorrection(values) {
	if (!values?.labScale || !values?.labOffset) return false;
	const isIdentityScale = values.labScale.every((v) => Math.abs(v - 1) < 0.001);
	const isIdentityOffset = values.labOffset.every((v) => Math.abs(v) < 0.1);
	return !(isIdentityScale && isIdentityOffset);
}

/**
 * Compact LAB values display
 */
function LabBadge({ values, side }) {
	const hasLab = hasLabCorrection(values);
	if (!hasLab) return null;

	return (
		<div className="text-xs text-muted-foreground bg-muted/50 rounded px-2 py-1">
			<span className="font-medium text-foreground">{side}:</span>{' '}
			<span className="font-mono">
				L×{values.labScale[0].toFixed(2)} a×{values.labScale[1].toFixed(2)} b×{values.labScale[2].toFixed(2)}
			</span>
		</div>
	);
}

/**
 * Simplified Color Correction Panel
 */
export function DualColorCorrectionPanel({
	leftValues,
	rightValues,
	onLeftChange,
	onRightChange,
	onResetAll,
	matchId,
	videoRef,
}) {
	const [autoLoading, setAutoLoading] = useState(false);
	const [isCollapsed, setIsCollapsed] = useState(true);
	const [showAdvanced, setShowAdvanced] = useState(false);

	const hasAnyCorrection = hasLabCorrection(leftValues) || hasLabCorrection(rightValues);

	const handleAutoColorCorrection = useCallback(async () => {
		if (!matchId) return;
		setAutoLoading(true);
		try {
			const videoElement = videoRef?.current ?? videoRef;
			const currentTime = videoElement?.currentTime || 0;
			const result = await autoColorCorrection(matchId, currentTime);
			if (result?.colorCorrection) {
				if (result.colorCorrection.left) {
					onLeftChange({ ...DEFAULT_COLOR_CORRECTION, ...result.colorCorrection.left });
				}
				if (result.colorCorrection.right) {
					onRightChange({ ...DEFAULT_COLOR_CORRECTION, ...result.colorCorrection.right });
				}
			}
		} catch (err) {
			console.error('Auto color correction failed:', err);
		} finally {
			setAutoLoading(false);
		}
	}, [matchId, videoRef, onLeftChange, onRightChange]);

	const handleReset = useCallback(() => {
		onResetAll();
	}, [onResetAll]);

	// Advanced slider handler
	const handleSliderChange = useCallback(
		(side, key, value) => {
			const setter = side === 'left' ? onLeftChange : onRightChange;
			const current = side === 'left' ? leftValues : rightValues;
			setter({ ...current, [key]: value });
		},
		[leftValues, rightValues, onLeftChange, onRightChange]
	);

	return (
		<div className="bg-card border rounded-lg shadow-sm overflow-hidden">
			{/* Header */}
			<button
				onClick={() => setIsCollapsed(!isCollapsed)}
				className="w-full flex items-center justify-between px-4 py-3 hover:bg-muted/20 transition-colors"
			>
				<div className="flex items-center gap-3">
					<Palette className="h-4 w-4 text-muted-foreground" />
					<span className="font-medium">Color Match</span>
					{hasAnyCorrection && (
						<span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30 px-2 py-0.5 rounded-full">
							<Check className="h-3 w-3" />
							Applied
						</span>
					)}
				</div>
				<ChevronDown
					className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`}
				/>
			</button>

			{/* Content */}
			{!isCollapsed && (
				<div className="px-4 pb-4 space-y-4">
					{/* Main Actions Row - Auto Color + Seam Blend inline */}
					<div className="flex items-center gap-3 pt-1">
						<Button
							onClick={handleAutoColorCorrection}
							disabled={autoLoading || !matchId}
							size="sm"
							className="shrink-0"
						>
							<Wand2 className={`h-4 w-4 mr-2 ${autoLoading ? 'animate-spin' : ''}`} />
							{autoLoading ? 'Analyzing...' : 'Auto Color'}
						</Button>
						{hasAnyCorrection && (
							<Button onClick={handleReset} variant="outline" size="sm" className="shrink-0">
								<RotateCcw className="h-4 w-4" />
							</Button>
						)}
						<div className="flex-1 flex items-center gap-2 min-w-0">
							<Blend className="h-4 w-4 text-muted-foreground shrink-0" />
							<span className="text-xs text-muted-foreground shrink-0">Blend</span>
							<InlineBlendSlider />
						</div>
					</div>

					{hasAnyCorrection ? (
						<div className="flex items-center gap-2 flex-wrap">
							<LabBadge values={rightValues} side="Right cam" />
						</div>
					) : (
						<p className="text-xs text-muted-foreground">
							Seek to a frame with similar content (grass), then click Auto Color.
						</p>
					)}

					{/* Advanced Toggle */}
					<button
						onClick={() => setShowAdvanced(!showAdvanced)}
						className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
					>
						<Settings2 className="h-3 w-3" />
						{showAdvanced ? 'Hide' : 'Show'} manual adjustments
						<ChevronDown className={`h-3 w-3 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
					</button>

					{/* Advanced Manual Controls */}
					{showAdvanced && (
						<div className="space-y-4 pt-2 border-t">
							<p className="text-xs text-muted-foreground">
								Fine-tune the color correction. These adjustments apply on top of Auto Match.
							</p>
							<div className="grid grid-cols-2 gap-6">
								{/* Left Camera */}
								<div className="space-y-3">
									<h5 className="text-xs font-medium text-muted-foreground">Left Camera</h5>
									<SliderControl
										label="Brightness"
										value={leftValues.brightness}
										min={-0.5}
										max={0.5}
										step={0.01}
										format={(v) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('left', 'brightness', v)}
									/>
									<SliderControl
										label="Contrast"
										value={leftValues.contrast}
										min={0.5}
										max={1.5}
										step={0.01}
										format={(v) => `${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('left', 'contrast', v)}
									/>
									<SliderControl
										label="Saturation"
										value={leftValues.saturation}
										min={0}
										max={2}
										step={0.01}
										format={(v) => `${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('left', 'saturation', v)}
									/>
								</div>

								{/* Right Camera */}
								<div className="space-y-3">
									<h5 className="text-xs font-medium text-muted-foreground">Right Camera</h5>
									<SliderControl
										label="Brightness"
										value={rightValues.brightness}
										min={-0.5}
										max={0.5}
										step={0.01}
										format={(v) => `${v > 0 ? '+' : ''}${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('right', 'brightness', v)}
									/>
									<SliderControl
										label="Contrast"
										value={rightValues.contrast}
										min={0.5}
										max={1.5}
										step={0.01}
										format={(v) => `${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('right', 'contrast', v)}
									/>
									<SliderControl
										label="Saturation"
										value={rightValues.saturation}
										min={0}
										max={2}
										step={0.01}
										format={(v) => `${(v * 100).toFixed(0)}%`}
										onChange={(v) => handleSliderChange('right', 'saturation', v)}
									/>
								</div>
							</div>
						</div>
					)}
				</div>
			)}
		</div>
	);
}

/**
 * Inline blend slider (compact)
 */
function InlineBlendSlider() {
	const blendWidth = useViewerStore((s) => s.blendWidth);
	const setBlendWidth = useViewerStore((s) => s.setBlendWidth);

	return (
		<div className="flex items-center gap-2 flex-1 min-w-0">
			<Slider
				min={0}
				max={50}
				step={1}
				value={[blendWidth * 100]}
				onValueChange={([v]) => setBlendWidth(v / 100)}
				className="flex-1"
			/>
			<span className="text-xs text-muted-foreground font-mono w-8 text-right">
				{blendWidth === 0 ? 'Off' : `${Math.round(blendWidth * 100)}%`}
			</span>
		</div>
	);
}

/**
 * Simple slider control
 */
function SliderControl({ label, value, min, max, step, format, onChange }) {
	return (
		<div className="space-y-1">
			<div className="flex items-center justify-between">
				<Label className="text-xs">{label}</Label>
				<span className="text-xs text-muted-foreground font-mono">{format(value)}</span>
			</div>
			<Slider
				min={min * 100}
				max={max * 100}
				step={step * 100}
				value={[value * 100]}
				onValueChange={([v]) => onChange(v / 100)}
			/>
		</div>
	);
}

export default DualColorCorrectionPanel;
