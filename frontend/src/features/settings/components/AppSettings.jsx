import React, { useEffect, useState } from 'react';
import { useSettings } from '@/hooks/useSettings';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
	Trash2,
	RotateCcw,
	Settings2,
	Database,
	Bug,
	ChevronDown,
	ChevronUp,
	FolderOpen,
	RefreshCw,
	Video,
} from 'lucide-react';
import { getEncoderSettings, updateEncoderSettings } from '@/features/settings/api/settings';

export default function AppSettings() {
	const { settings, updateSetting, resetSettings } = useSettings();
	const [encoderInfo, setEncoderInfo] = useState(null);
	const [loadingEncoderInfo, setLoadingEncoderInfo] = useState(false);
	const [encoderError, setEncoderError] = useState(null);
	const [updatingEncoder, setUpdatingEncoder] = useState(false);
	const [telemetryUploadStatus, setTelemetryUploadStatus] = useState(null);
	const [telemetryUploading, setTelemetryUploading] = useState(false);
	const [showTelemetryDetails, setShowTelemetryDetails] = useState(false);

	const loadEncoderInfo = async () => {
		setLoadingEncoderInfo(true);
		setEncoderError(null);
		try {
			const info = await getEncoderSettings();
			setEncoderInfo(info);
		} catch (err) {
			setEncoderError(err?.message || 'Failed to load encoder settings');
		} finally {
			setLoadingEncoderInfo(false);
		}
	};

	const handleEncoderChange = async (encoder) => {
		if (!encoder) return;
		setUpdatingEncoder(true);
		setEncoderError(null);
		try {
			await updateEncoderSettings(encoder);
			await loadEncoderInfo();
		} catch (err) {
			setEncoderError(err?.message || 'Failed to update encoder settings');
		} finally {
			setUpdatingEncoder(false);
		}
	};

	useEffect(() => {
		loadEncoderInfo();
	}, []);

	const handleClearStorage = (key, label) => {
		try {
			if (key === 'all') {
				localStorage.clear();
				window.location.reload();
			} else {
				localStorage.removeItem(key);
				alert(`${label} cleared successfully`);
			}
		} catch (err) {
			alert(`Failed to clear ${label}: ${err.message}`);
		}
	};

	const handleClearUserData = async () => {
		if (!window.electronAPI?.clearUserDataFolder) {
			alert('This feature is only available in the desktop app');
			return;
		}

		const result = await window.electronAPI.clearUserDataFolder();
		if (!result.ok && !result.cancelled) {
			alert(`Failed to clear user data: ${result.error || 'Unknown error'}`);
		}
		// If successful, the app will quit automatically
	};

	return (
		<div className="space-y-6">
			{/* Encoder Settings */}
			<Card id="encoder">
				<CardHeader>
					<div className="flex items-center gap-2">
						<Settings2 className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Encoder</CardTitle>
					</div>
					<CardDescription>
						Choose which encoder FFmpeg should prefer for transcoding. Auto picks the best available.
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="flex items-center justify-between gap-3">
						<div className="space-y-1">
							<Label className="text-base">Preferred encoder</Label>
							<p className="text-sm text-muted-foreground">
								Stored in the backend settings. Hardware options only appear if available.
							</p>
						</div>
						<Button
							variant="outline"
							size="sm"
							onClick={loadEncoderInfo}
							disabled={loadingEncoderInfo || updatingEncoder}
						>
							Refresh
						</Button>
					</div>

					<div className="flex items-center gap-2">
						<Select
							value={encoderInfo?.current_encoder || ''}
							onValueChange={handleEncoderChange}
							disabled={!encoderInfo || loadingEncoderInfo || updatingEncoder}
						>
							<SelectTrigger className="w-full max-w-sm">
								<SelectValue placeholder={loadingEncoderInfo ? 'Loading…' : 'Select encoder'} />
							</SelectTrigger>
							<SelectContent>
								{encoderInfo?.available_encoders?.map((enc) => (
									<SelectItem key={enc} value={enc}>
										{encoderInfo?.encoder_descriptions?.[enc] || enc}
									</SelectItem>
								))}
							</SelectContent>
						</Select>

						{encoderInfo?.current_encoder && (
							<Badge variant={encoderInfo.current_encoder === 'libx264' ? 'secondary' : 'default'}>
								{encoderInfo.current_encoder}
							</Badge>
						)}
					</div>

					{encoderInfo?.current_encoder && (
						<p className="text-xs text-muted-foreground">
							{encoderInfo?.encoder_descriptions?.[encoderInfo.current_encoder]}
						</p>
					)}

					{encoderError && <p className="text-sm text-destructive">{encoderError}</p>}
				</CardContent>
			</Card>
			{/* Recording Settings */}
			<Card id="recording">
				<CardHeader>
					<div className="flex items-center gap-2">
						<Video className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Recording</CardTitle>
						<span className="px-2 py-0.5 text-xs font-medium bg-yellow-500/10 text-yellow-600 dark:text-yellow-500 rounded-full border border-yellow-500/20">
							Experimental
						</span>
					</div>
					<CardDescription>
						Configure canvas recording quality and format for sharing clips. This feature is experimental
						and may have issues with certain browsers or configurations.
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="space-y-2">
						<Label htmlFor="recording-bitrate">Quality (Bitrate)</Label>
						<Select
							value={String(settings.recordingBitrate ?? 16)}
							onValueChange={(value) => updateSetting('recordingBitrate', parseInt(value))}
						>
							<SelectTrigger id="recording-bitrate">
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="8">8 Mbps (Small file)</SelectItem>
								<SelectItem value="16">16 Mbps (Balanced)</SelectItem>
								<SelectItem value="30">30 Mbps (High quality)</SelectItem>
								<SelectItem value="45">45 Mbps (Very high)</SelectItem>
								<SelectItem value="60">60 Mbps (Maximum)</SelectItem>
							</SelectContent>
						</Select>
						<p className="text-xs text-muted-foreground">
							Higher bitrate = better quality, larger file size
						</p>
					</div>

					<Separator />

					<div className="space-y-2">
						<Label htmlFor="recording-format">Output Format</Label>
						<Select
							value={settings.recordingFormat ?? 'webm'}
							onValueChange={(value) => updateSetting('recordingFormat', value)}
						>
							<SelectTrigger id="recording-format" className="w-full max-w-sm">
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="webm">WebM (VP9) - Best compatibility</SelectItem>
								<SelectItem value="webm-vp8">WebM (VP8) - Wider support</SelectItem>
							</SelectContent>
						</Select>
						<p className="text-xs text-muted-foreground">
							WebM format is supported by most browsers and video players
						</p>
					</div>

					<div className="bg-muted/50 p-3 rounded-md space-y-2">
						<p className="text-xs text-muted-foreground">
							💡 Press <kbd className="px-1 py-0.5 bg-background rounded text-xs font-mono">R</kbd> in the
							viewer to start/stop recording.
						</p>
						<p className="text-xs text-muted-foreground">
							🖥️ <strong>Fullscreen recommended:</strong> Recording in fullscreen captures at your
							screen&apos;s native resolution for the best quality.
						</p>
					</div>
				</CardContent>
			</Card>
			{/* Telemetry */}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Database className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Telemetry</CardTitle>
					</div>
					<CardDescription>
						Optional anonymous usage data to help improve the app. No personal data or file content
						collected.
					</CardDescription>
				</CardHeader>
				<CardContent className="space-y-6">
					<button
						onClick={() => setShowTelemetryDetails(!showTelemetryDetails)}
						className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
					>
						{showTelemetryDetails ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
						{showTelemetryDetails ? 'Hide details' : 'What data is collected?'}
					</button>

					{showTelemetryDetails && (
						<div className="space-y-2 text-sm pb-4">
							<div className="bg-muted/50 p-3 rounded-md space-y-2">
								<p className="font-medium">What we collect:</p>
								<ul className="list-disc list-inside space-y-1 text-muted-foreground ml-2">
									<li>Feature usage (pages visited, buttons clicked)</li>
									<li>Performance metrics (processing times, errors)</li>
									<li>Basic system info (OS, hardware) — optional</li>
								</ul>
							</div>
							<div className="bg-green-500/10 border border-green-500/20 p-3 rounded-md">
								<p className="font-medium text-green-700 dark:text-green-400 mb-1">🔒 Privacy-first</p>
								<ul className="list-disc list-inside space-y-1 text-muted-foreground ml-2">
									<li>Anonymous, no tracking</li>
									<li>Never sold or shared</li>
									<li>Stored locally, optional upload</li>
								</ul>
							</div>
						</div>
					)}
					<div className="flex items-center justify-between space-x-4">
						<div className="flex-1 space-y-1">
							<Label htmlFor="telemetry-enabled" className="text-base cursor-pointer">
								Enable telemetry
							</Label>
							<p className="text-sm text-muted-foreground">
								Collect anonymous usage data locally (e.g., app_open, match_created, errors).
							</p>
						</div>
						<Switch
							id="telemetry-enabled"
							checked={!!settings.telemetryEnabled}
							onCheckedChange={(checked) => {
								updateSetting('telemetryEnabled', checked);
								if (!checked) {
									updateSetting('telemetryIncludeSystemInfo', false);
									updateSetting('telemetryAutoUpload', false);
								}
							}}
						/>
					</div>

					<div className="flex items-center justify-between space-x-4">
						<div className="flex-1 space-y-1">
							<Label htmlFor="telemetry-system" className="text-base cursor-pointer">
								Include system info
							</Label>
							<p className="text-sm text-muted-foreground">
								Add OS, CPU, RAM, and GPU info to help diagnose hardware-specific issues.
							</p>
						</div>
						<Switch
							id="telemetry-system"
							disabled={!settings.telemetryEnabled}
							checked={!!settings.telemetryIncludeSystemInfo}
							onCheckedChange={(checked) => updateSetting('telemetryIncludeSystemInfo', checked)}
						/>
					</div>

					<div className="flex items-center justify-between space-x-4">
						<div className="flex-1 space-y-1">
							<Label htmlFor="telemetry-auto-upload" className="text-base cursor-pointer">
								Automatic upload
							</Label>
							<p className="text-sm text-muted-foreground">
								Automatically upload telemetry data to the remote endpoint every 5 minutes.
							</p>
						</div>
						<Switch
							id="telemetry-auto-upload"
							checked={settings.telemetryAutoUpload}
							onCheckedChange={async (checked) => {
								updateSetting('telemetryAutoUpload', checked);
								if (
									checked &&
									window.electronAPI?.telemetryUploadNow &&
									settings.telemetryEndpointUrl
								) {
									try {
										await window.electronAPI.telemetryUploadNow({
											endpointUrl: settings.telemetryEndpointUrl,
										});
									} catch (err) {
										console.error('Auto-upload activation upload failed:', err);
									}
								}
							}}
							disabled={!settings.telemetryEnabled || !settings.telemetryEndpointUrl?.trim()}
						/>
					</div>

					<div className="space-y-2">
						<Label htmlFor="telemetry-endpoint" className="text-base">
							Endpoint URL
						</Label>
						<p className="text-xs text-muted-foreground">Remote server endpoint for telemetry uploads.</p>
						<div className="flex gap-2">
							<Input
								id="telemetry-endpoint"
								type="url"
								value={settings.telemetryEndpointUrl}
								onChange={(e) => updateSetting('telemetryEndpointUrl', e.target.value)}
								placeholder="https://your-domain.com/telemetry"
								className="font-mono text-sm"
								disabled={!settings.telemetryEnabled}
							/>
							<Button
								variant="outline"
								size="sm"
								disabled={
									!settings.telemetryEnabled ||
									telemetryUploading ||
									!settings.telemetryEndpointUrl?.trim()
								}
								onClick={async () => {
									setTelemetryUploadStatus(null);
									if (!window.electronAPI?.telemetryUploadNow) {
										setTelemetryUploadStatus({
											ok: false,
											error: 'Upload is only available in the desktop app.',
										});
										return;
									}

									setTelemetryUploading(true);
									try {
										const res = await window.electronAPI.telemetryUploadNow({
											endpointUrl: settings.telemetryEndpointUrl,
										});
										setTelemetryUploadStatus(res);
									} catch (err) {
										setTelemetryUploadStatus({ ok: false, error: err?.message || 'Upload failed' });
									} finally {
										setTelemetryUploading(false);
									}
								}}
							>
								{telemetryUploading ? 'Uploading…' : 'Upload manually'}
							</Button>
						</div>
						{telemetryUploadStatus && (
							<p className="text-xs">
								{telemetryUploadStatus.ok
									? `Uploaded ${telemetryUploadStatus.sent || 0} line(s). Remaining: ${telemetryUploadStatus.remaining_lines ?? 0}.`
									: `Upload failed: ${telemetryUploadStatus.error || 'Unknown error'}`}
							</p>
						)}
					</div>

					<Separator />

					<div className="space-y-2">
						<Label className="text-base">Local storage</Label>
						<p className="text-sm text-muted-foreground">
							All events are saved locally on your machine. You can view the files anytime.
						</p>
						<div className="flex gap-2 flex-wrap">
							<Button
								variant="outline"
								size="sm"
								onClick={async () => {
									if (window.electronAPI?.telemetryOpenFolder) {
										await window.electronAPI.telemetryOpenFolder();
									} else {
										alert('Telemetry folder is only available in the desktop app.');
									}
								}}
							>
								<FolderOpen className="h-3 w-3 mr-2" />
								Open folder
							</Button>
							<Button
								variant="outline"
								size="sm"
								onClick={async () => {
									if (!window.electronAPI?.telemetryDeleteLocal) return;

									if (
										!confirm(
											'Delete all local telemetry data? This cannot be undone.\n\nTo delete online telemetry, please email the developer.'
										)
									) {
										return;
									}

									try {
										const res = await window.electronAPI.telemetryDeleteLocal();
										if (res.ok) {
											alert('Local telemetry data deleted successfully.');
										} else {
											alert(`Failed to delete: ${res.error || 'Unknown error'}`);
										}
									} catch (err) {
										alert(`Failed to delete: ${err?.message || 'Unknown error'}`);
									}
								}}
							>
								<Trash2 className="h-3 w-3 mr-2" />
								Delete local telemetry
							</Button>
							<Button
								variant="outline"
								size="sm"
								onClick={async () => {
									if (!window.electronAPI?.telemetryResetClientId) return;

									if (
										!confirm(
											'Reset client ID? A new anonymous ID will be generated.\n\nThis is useful if you want to start fresh with a new identity.'
										)
									) {
										return;
									}

									try {
										const res = await window.electronAPI.telemetryResetClientId();
										if (res.ok) {
											alert(`Client ID reset successfully.\n\nNew ID: ${res.client_id}`);
										} else {
											alert(`Failed to reset: ${res.error || 'Unknown error'}`);
										}
									} catch (err) {
										alert(`Failed to reset: ${err?.message || 'Unknown error'}`);
									}
								}}
							>
								<RefreshCw className="h-3 w-3 mr-2" />
								Reset client ID
							</Button>
						</div>
					</div>
				</CardContent>
			</Card>
			{/* Developer Settings */}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Bug className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Developer</CardTitle>
					</div>
					<CardDescription>Debug and troubleshooting options</CardDescription>
				</CardHeader>
				<CardContent className="space-y-6">
					<div className="flex items-center justify-between space-x-4">
						<div className="flex-1 space-y-1">
							<Label htmlFor="debug-mode" className="text-base cursor-pointer">
								Debug Mode
							</Label>
							<p className="text-sm text-muted-foreground">
								Save debug frames to temp folder, show timing metrics, and enable verbose logging
							</p>
						</div>
						<Switch
							id="debug-mode"
							checked={settings.debugMode}
							onCheckedChange={(checked) => updateSetting('debugMode', checked)}
						/>
					</div>
				</CardContent>
			</Card>
			{/* Connection Settings */}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Settings2 className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Connection</CardTitle>
					</div>
					<CardDescription>Configure backend server connection</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="space-y-2">
						<Label htmlFor="api-url">Backend API URL</Label>
						<div className="flex gap-2">
							<Input
								id="api-url"
								type="url"
								value={settings.apiBaseUrl}
								onChange={(e) => updateSetting('apiBaseUrl', e.target.value)}
								placeholder="http://127.0.0.1:8000/api"
								className="font-mono text-sm"
							/>
							<Button
								variant="ghost"
								size="icon"
								onClick={() => updateSetting('apiBaseUrl', 'http://127.0.0.1:8000/api')}
								title="Reset to default"
							>
								<RotateCcw className="h-4 w-4" />
							</Button>
						</div>
						<p className="text-xs text-muted-foreground">Change requires page reload to take effect</p>
					</div>
				</CardContent>
			</Card>
			{/* Display Settings */}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Settings2 className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Display</CardTitle>
					</div>
					<CardDescription>Graphics rendering and display options</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="flex items-center justify-between space-x-4">
						<div className="flex-1 space-y-1">
							<Label htmlFor="disable-hw-accel" className="text-base cursor-pointer">
								Disable hardware acceleration
							</Label>
							<p className="text-sm text-muted-foreground">
								Fixes font aliasing issues with NVIDIA FXAA and some GPU drivers. Requires app restart.
							</p>
						</div>
						<Switch
							id="disable-hw-accel"
							checked={settings.disableHardwareAcceleration ?? false}
							onCheckedChange={(checked) => updateSetting('disableHardwareAcceleration', checked)}
						/>
					</div>
					<div className="bg-muted/50 p-3 rounded-md">
						<p className="text-xs text-muted-foreground">
							💡 If text looks blurry or aliased, try enabling this option and restart the app.
						</p>
					</div>
				</CardContent>
			</Card>
			{/* Storage */}{' '}
			<Card>
				<CardHeader>
					<div className="flex items-center gap-2">
						<Database className="h-5 w-5 text-muted-foreground" />
						<CardTitle>Storage</CardTitle>
					</div>
					<CardDescription>Manage local data and cache</CardDescription>
				</CardHeader>
				<CardContent className="space-y-4">
					<div className="flex items-center justify-between">
						<div className="flex-1">
							<Label className="text-base">Clear Draft Matches</Label>
							<p className="text-sm text-muted-foreground">Remove saved wizard progress</p>
						</div>
						<Button
							variant="outline"
							size="sm"
							onClick={() => handleClearStorage('match-wizard-draft', 'Draft matches')}
						>
							<Trash2 className="h-4 w-4 mr-2" />
							Clear
						</Button>
					</div>

					<Separator />

					<div className="flex items-center justify-between">
						<div className="flex-1">
							<Label className="text-base">User Data Folder</Label>
							<p className="text-sm text-muted-foreground">
								Open the folder containing all app data and settings
							</p>
						</div>
						<Button
							variant="outline"
							size="sm"
							onClick={async () => {
								if (!window.electronAPI?.openUserDataFolder) return;
								await window.electronAPI.openUserDataFolder();
							}}
						>
							<FolderOpen className="h-4 w-4 mr-2" />
							Open folder
						</Button>
					</div>

					<Separator />

					<div className="flex items-center justify-between">
						<div className="flex-1">
							<Label className="text-base">Reset All Settings</Label>
							<p className="text-sm text-muted-foreground">Restore all settings to default values</p>
						</div>
						<Button variant="outline" size="sm" onClick={resetSettings}>
							<RotateCcw className="h-4 w-4 mr-2" />
							Reset
						</Button>
					</div>

					<Separator />

					<div className="flex items-center justify-between">
						<div className="flex-1">
							<div className="flex items-center gap-2">
								<Label className="text-base text-destructive">Clear UI Cache</Label>
								<Badge variant="destructive" className="text-xs">
									Destructive
								</Badge>
							</div>
							<p className="text-sm text-muted-foreground">
								Clear browser cache and UI state (drafts, viewing history). Does not delete matches,
								videos, or settings.
							</p>
						</div>
						<Button variant="destructive" size="sm" onClick={() => handleClearStorage('all', 'UI cache')}>
							<Trash2 className="h-4 w-4 mr-2" />
							Clear Cache
						</Button>
					</div>

					<Separator />

					<div className="flex items-center justify-between">
						<div className="flex-1">
							<div className="flex items-center gap-2">
								<Label className="text-base text-destructive">Delete All User Data</Label>
								<Badge variant="destructive" className="text-xs font-bold">
									NUCLEAR OPTION
								</Badge>
							</div>
							<p className="text-sm text-muted-foreground">
								Permanently delete ALL data: matches, videos, settings, telemetry, logs. App will quit.
								Cannot be undone.
							</p>
						</div>
						<Button variant="destructive" size="sm" onClick={handleClearUserData}>
							<Trash2 className="h-4 w-4 mr-2" />
							Delete Everything
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
