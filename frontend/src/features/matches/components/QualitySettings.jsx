import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Settings2, Zap } from 'lucide-react';

const PRESET_DESCRIPTIONS = {
	'720p': {
		title: '720p HD',
		specs: '30 Mbps • 1280x1440 stacked',
		description: 'Good for quick previews and testing',
	},
	'1080p': {
		title: '1080p Full HD (Recommended)',
		specs: '50 Mbps • 1920x2160 stacked',
		description: 'Balanced quality and file size',
	},
	'1440p': {
		title: '1440p QHD',
		specs: '70 Mbps • 2560x2880 stacked',
		description: 'High quality for detailed calibration',
	},
};

function PresetDescription({ preset }) {
	const info = PRESET_DESCRIPTIONS[preset];
	if (!info) return null;

	return (
		<div className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md">
			<div className="font-medium mb-1">{info.title}</div>
			<div>{info.specs}</div>
			<div className="text-xs mt-1">{info.description}</div>
		</div>
	);
}

// TODO: Could possibly use a setter instead
function CustomSettings({ bitrate, preset, resolution, useGpuDecode, onChange }) {
	return (
		<div className="space-y-3 border-t pt-3 mt-2">
			{/* Bitrate */}
			<div className="space-y-2">
				<Label className="text-sm">Bitrate</Label>
				<Select value={bitrate} onValueChange={(v) => onChange({ bitrate: v })}>
					<SelectTrigger className="h-9">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="20M">20 Mbps (Low)</SelectItem>
						<SelectItem value="30M">30 Mbps (Medium)</SelectItem>
						<SelectItem value="40M">40 Mbps (High)</SelectItem>
						<SelectItem value="50M">50 Mbps (Very High)</SelectItem>
						<SelectItem value="70M">70 Mbps (Ultra)</SelectItem>
						<SelectItem value="90M">90 Mbps (Extreme)</SelectItem>
						<SelectItem value="120M">120 Mbps (Max)</SelectItem>
					</SelectContent>
				</Select>
				<div className="text-xs text-muted-foreground">Higher = better quality, larger file</div>
			</div>

			{/* Speed Preset */}
			<div className="space-y-2">
				<Label className="text-sm">Speed Preset</Label>
				<Select value={preset} onValueChange={(v) => onChange({ preset: v })}>
					<SelectTrigger className="h-9">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="ultrafast">Ultra Fast</SelectItem>
						<SelectItem value="superfast">Super Fast</SelectItem>
						<SelectItem value="veryfast">Very Fast</SelectItem>
						<SelectItem value="faster">Faster</SelectItem>
						<SelectItem value="fast">Fast</SelectItem>
						<SelectItem value="medium">Medium</SelectItem>
						<SelectItem value="slow">Slow</SelectItem>
						<SelectItem value="slower">Slower</SelectItem>
					</SelectContent>
				</Select>
				<div className="text-xs text-muted-foreground">
					Faster = quicker encoding, less compression. GPU presets auto-mapped.
				</div>
			</div>

			{/* Resolution */}
			<div className="space-y-2">
				<Label className="text-sm">Output Resolution</Label>
				<Select value={resolution} onValueChange={(v) => onChange({ resolution: v })}>
					<SelectTrigger className="h-9">
						<SelectValue />
					</SelectTrigger>
					<SelectContent>
						<SelectItem value="720p">720p (1280x1440 stacked)</SelectItem>
						<SelectItem value="1080p">1080p (1920x2160 stacked)</SelectItem>
						<SelectItem value="1440p">1440p (2560x2880 stacked)</SelectItem>
						<SelectItem value="4k">4K (3840x4320 stacked) - H.265 recommended</SelectItem>
					</SelectContent>
				</Select>
				<div className="text-xs text-muted-foreground">
					Higher = better quality, larger file, slower processing
				</div>
			</div>

			{/* GPU Decode */}
			<div className="space-y-2">
				<div className="flex items-center gap-2">
					<input
						type="checkbox"
						id="gpu-decode"
						checked={useGpuDecode}
						onChange={(e) => onChange({ useGpuDecode: e.target.checked })}
						className="w-4 h-4 rounded"
					/>
					<Label htmlFor="gpu-decode" className="text-sm cursor-pointer">
						Use GPU decoding
					</Label>
				</div>
				<div className="text-xs text-muted-foreground">May be faster or slower depending on your hardware</div>
			</div>
		</div>
	);
}

// TODO: Could compute the encoder info by itself rather than passing from parent
function EncoderInfo({ encoderInfo, loading }) {
	return (
		<div className="border-t pt-4 mt-4">
			<div className="flex items-center justify-between gap-4">
				<div className="flex items-center gap-3 flex-1">
					<Zap className="h-4 w-4" />
					<div className="flex items-center gap-2 flex-wrap text-sm">
						<span className="font-semibold">Video Encoder:</span>
						{loading ? (
							<span className="text-muted-foreground">Loading...</span>
						) : encoderInfo ? (
							<>
								<Badge
									variant={encoderInfo.current_encoder === 'libx264' ? 'secondary' : 'default'}
									className="gap-1"
								>
									{encoderInfo.encoder_descriptions[encoderInfo.current_encoder]}
								</Badge>
								{encoderInfo.current_encoder === 'libx264' && (
									<span className="text-muted-foreground text-xs">(Slower)</span>
								)}
							</>
						) : (
							<span className="text-muted-foreground">Unknown</span>
						)}
					</div>
				</div>
				<Link to="/profiles?tab=settings#encoder">
					<Button variant="outline" size="sm" className="gap-2">
						<Settings2 className="h-3 w-3" />
						Change
					</Button>
				</Link>
			</div>
		</div>
	);
}

// TODO: Should not need so many props. Just setter for a quality settings object
export default function QualitySettings({
	qualityPreset,
	onPresetChange,
	customBitrate,
	customPreset,
	customResolution,
	customUseGpuDecode,
	onCustomChange,
	encoderInfo,
	loadingEncoder,
}) {
	return (
		<Card>
			<CardHeader>
				<CardTitle>Processing Quality</CardTitle>
			</CardHeader>
			<CardContent className="space-y-4">
				{/* Quality Preset Dropdown */}
				<div className="space-y-2">
					<Label>Quality Preset</Label>
					<Select value={qualityPreset} onValueChange={onPresetChange}>
						<SelectTrigger>
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							<SelectItem value="720p">720p HD</SelectItem>
							<SelectItem value="1080p">1080p Full HD (Recommended)</SelectItem>
							<SelectItem value="1440p">1440p QHD</SelectItem>
							<SelectItem value="custom">⚙️ Custom</SelectItem>
						</SelectContent>
					</Select>

					{/* Preset Description or Custom Settings */}
					{qualityPreset !== 'custom' ? (
						<PresetDescription preset={qualityPreset} />
					) : (
						<CustomSettings
							bitrate={customBitrate}
							preset={customPreset}
							resolution={customResolution}
							useGpuDecode={customUseGpuDecode}
							onChange={onCustomChange}
						/>
					)}
				</div>

				<EncoderInfo encoderInfo={encoderInfo} loading={loadingEncoder} />
			</CardContent>
		</Card>
	);
}
