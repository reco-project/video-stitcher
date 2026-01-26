import React, { useState, useEffect, useCallback } from 'react';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Palette, RotateCcw, Sun, Contrast, Droplets, Thermometer, Wand2, ChevronDown } from 'lucide-react';
import { DEFAULT_COLOR_CORRECTION } from '../utils/utils';
import { autoColorCorrection } from '@/features/matches/api/matches.js';

/**
 * ColorCorrectionPanel - Controls for adjusting color correction per camera
 */
export default function ColorCorrectionPanel({ side, values, onChange, onReset }) {
	const sideLabel = side === 'left' ? 'Left' : 'Right';

	const handleChange = useCallback(
		(key, value) => {
			onChange({ ...values, [key]: value });
		},
		[values, onChange]
	);

	const handleColorBalanceChange = useCallback(
		(index, value) => {
			const newBalance = [...values.colorBalance];
			newBalance[index] = value;
			onChange({ ...values, colorBalance: newBalance });
		},
		[values, onChange]
	);

	return (
		<div className="space-y-3">
			<div className="flex items-center justify-between">
				<h5 className="text-xs font-medium flex items-center gap-1.5">
					<Palette className="h-3 w-3" />
					{sideLabel} Camera
				</h5>
				<Button
					type="button"
					variant="ghost"
					size="sm"
					onClick={onReset}
					className="h-6 px-2 text-xs"
					title="Reset to defaults"
				>
					<RotateCcw className="h-3 w-3" />
				</Button>
			</div>

			{/* Brightness */}
			<div className="space-y-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs flex items-center gap-1">
						<Sun className="h-3 w-3" />
						Brightness
					</Label>
					<span className="text-xs text-muted-foreground w-12 text-right">
						{values.brightness > 0 ? '+' : ''}
						{(values.brightness * 100).toFixed(0)}%
					</span>
				</div>
				<Slider
					min={-50}
					max={50}
					step={1}
					value={[values.brightness * 100]}
					onValueChange={([v]) => handleChange('brightness', v / 100)}
				/>
			</div>

			{/* Contrast */}
			<div className="space-y-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs flex items-center gap-1">
						<Contrast className="h-3 w-3" />
						Contrast
					</Label>
					<span className="text-xs text-muted-foreground w-12 text-right">
						{(values.contrast * 100).toFixed(0)}%
					</span>
				</div>
				<Slider
					min={50}
					max={150}
					step={1}
					value={[values.contrast * 100]}
					onValueChange={([v]) => handleChange('contrast', v / 100)}
				/>
			</div>

			{/* Saturation */}
			<div className="space-y-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs flex items-center gap-1">
						<Droplets className="h-3 w-3" />
						Saturation
					</Label>
					<span className="text-xs text-muted-foreground w-12 text-right">
						{(values.saturation * 100).toFixed(0)}%
					</span>
				</div>
				<Slider
					min={0}
					max={200}
					step={1}
					value={[values.saturation * 100]}
					onValueChange={([v]) => handleChange('saturation', v / 100)}
				/>
			</div>

			{/* Temperature */}
			<div className="space-y-1">
				<div className="flex items-center justify-between">
					<Label className="text-xs flex items-center gap-1">
						<Thermometer className="h-3 w-3" />
						Temperature
					</Label>
					<span className="text-xs text-muted-foreground w-12 text-right">
						{values.temperature > 0 ? 'Warm' : values.temperature < 0 ? 'Cool' : 'Neutral'}
					</span>
				</div>
				<Slider
					min={-100}
					max={100}
					step={1}
					value={[values.temperature * 100]}
					onValueChange={([v]) => handleChange('temperature', v / 100)}
				/>
			</div>

			{/* RGB Color Balance */}
			<div className="space-y-2 pt-1">
				<Label className="text-xs">Color Balance (RGB)</Label>
				<div className="grid grid-cols-3 gap-2">
					{['R', 'G', 'B'].map((channel, index) => (
						<div key={channel} className="space-y-1">
							<div className="flex items-center justify-between">
								<span
									className={`text-xs font-medium ${
										index === 0 ? 'text-red-500' : index === 1 ? 'text-green-500' : 'text-blue-500'
									}`}
								>
									{channel}
								</span>
								<span className="text-xs text-muted-foreground">
									{(values.colorBalance[index] * 100).toFixed(0)}%
								</span>
							</div>
							<Slider
								min={50}
								max={150}
								step={1}
								value={[values.colorBalance[index] * 100]}
								onValueChange={([v]) => handleColorBalanceChange(index, v / 100)}
								className={
									index === 0
										? '[&_[role=slider]]:bg-red-500'
										: index === 1
											? '[&_[role=slider]]:bg-green-500'
											: '[&_[role=slider]]:bg-blue-500'
								}
							/>
						</div>
					))}
				</div>
			</div>
		</div>
	);
}

/**
 * Dual camera color correction panel with link/unlink option
 */
export function DualColorCorrectionPanel({ leftValues, rightValues, onLeftChange, onRightChange, onResetAll, matchId, videoRef }) {
	const [linked, setLinked] = useState(false);
	const [autoLoading, setAutoLoading] = useState(false);
	const [isCollapsed, setIsCollapsed] = useState(true);

	const handleLeftChange = useCallback(
		(newValues) => {
			onLeftChange(newValues);
			if (linked) {
				onRightChange(newValues);
			}
		},
		[linked, onLeftChange, onRightChange]
	);

	const handleRightChange = useCallback(
		(newValues) => {
			onRightChange(newValues);
			if (linked) {
				onLeftChange(newValues);
			}
		},
		[linked, onLeftChange, onRightChange]
	);

	const handleResetLeft = useCallback(() => {
		onLeftChange({ ...DEFAULT_COLOR_CORRECTION });
		if (linked) {
			onRightChange({ ...DEFAULT_COLOR_CORRECTION });
		}
	}, [linked, onLeftChange, onRightChange]);

	const handleResetRight = useCallback(() => {
		onRightChange({ ...DEFAULT_COLOR_CORRECTION });
		if (linked) {
			onLeftChange({ ...DEFAULT_COLOR_CORRECTION });
		}
	}, [linked, onLeftChange, onRightChange]);

	const handleAutoColorCorrection = useCallback(async () => {
		if (!matchId) return;
		setAutoLoading(true);
		try {
			// Get current video time when button is clicked
			// videoRef can be either a ref object (with .current) or the element directly
			const videoElement = videoRef?.current ?? videoRef;
			const currentTime = videoElement?.currentTime || 0;
			console.log('Auto color correction at time:', currentTime);
			const result = await autoColorCorrection(matchId, currentTime);
			console.log('Auto color correction result:', result);
			if (result?.colorCorrection) {
				if (result.colorCorrection.left) {
					// Merge with defaults to ensure all fields are present
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

	return (
		<div className="bg-card border rounded-lg shadow-sm overflow-hidden">
			{/* Collapsible Header - full width clickable like Info panel */}
			<button
				onClick={() => setIsCollapsed(!isCollapsed)}
				className="w-full flex items-center justify-between px-4 py-3 bg-muted/20 hover:bg-muted/40 transition-colors"
			>
				<div className="flex items-center gap-3">
					<h4 className="font-semibold flex items-center gap-2">
						<Palette className="h-4 w-4 text-muted-foreground" />
						Color Correction
					</h4>
				</div>
				<ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`} />
			</button>

			{/* Collapsible Content */}
			{!isCollapsed && (
				<>
					{/* Action buttons */}
					<div className="flex items-center gap-2 px-4 py-2 border-t bg-muted/10">
						{matchId && (
							<Button
								type="button"
								variant="outline"
								size="sm"
								onClick={handleAutoColorCorrection}
								disabled={autoLoading}
								className="h-6 px-2 text-xs"
								title="Auto-detect color correction from current frame (experimental)"
							>
								<Wand2 className={`h-3 w-3 mr-1 ${autoLoading ? 'animate-spin' : ''}`} />
								{autoLoading ? 'Analyzing...' : 'Auto ⚗️'}
							</Button>
						)}
						<Button
							type="button"
							variant={linked ? 'default' : 'outline'}
							size="sm"
							onClick={() => setLinked(!linked)}
							className="h-6 px-2 text-xs"
						>
							{linked ? '🔗 Linked' : '🔓 Independent'}
						</Button>
						<Button
							type="button"
							variant="ghost"
							size="sm"
							onClick={onResetAll}
							className="h-6 px-2 text-xs"
							title="Reset all to defaults"
						>
							<RotateCcw className="h-3 w-3 mr-1" />
							Reset
						</Button>
					</div>
					{/* Sliders */}
					<div className="px-4 pb-3 pt-2">
						<div className="grid grid-cols-2 gap-4">
							<ColorCorrectionPanel
								side="left"
								values={leftValues}
								onChange={handleLeftChange}
								onReset={handleResetLeft}
							/>
							<ColorCorrectionPanel
								side="right"
								values={rightValues}
								onChange={handleRightChange}
								onReset={handleResetRight}
							/>
						</div>
					</div>
				</>
			)}
		</div>
	);
}
