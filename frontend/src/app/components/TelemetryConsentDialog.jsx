import React, { useState, useEffect } from 'react';
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { useSettings } from '@/hooks/useSettings';
import { ChevronDown, ChevronUp } from 'lucide-react';

/**
 * First-run telemetry consent dialog
 * Appears once to encourage opt-in telemetry
 */
export default function TelemetryConsentDialog() {
	const { settings, updateSetting, loading } = useSettings();
	const [isOpen, setIsOpen] = useState(false);
	const [includeSystemInfo, setIncludeSystemInfo] = useState(true);
	const [showDetails, setShowDetails] = useState(false);
	const [hasBeenShown, setHasBeenShown] = useState(false);

	useEffect(() => {
		// Show dialog if user hasn't been prompted yet (check only once when loaded)
		if (!loading && settings && !settings.telemetryPromptShown && !hasBeenShown) {
			setIsOpen(true);
			setHasBeenShown(true);
		}
	}, [loading, settings.telemetryPromptShown, hasBeenShown]);

	const handleAccept = async () => {
		// Batch all updates together
		if (window.electronAPI?.writeSettings) {
			await window.electronAPI.writeSettings({
				...settings,
				telemetryEnabled: true,
				telemetryIncludeSystemInfo: includeSystemInfo,
				telemetryPromptShown: true,
			});
		}
		updateSetting('telemetryEnabled', true);
		updateSetting('telemetryIncludeSystemInfo', includeSystemInfo);
		updateSetting('telemetryPromptShown', true);
		setIsOpen(false);
	};

	const handleDecline = async () => {
		// Batch all updates together
		if (window.electronAPI?.writeSettings) {
			await window.electronAPI.writeSettings({
				...settings,
				telemetryEnabled: false,
				telemetryPromptShown: true,
			});
		}
		updateSetting('telemetryEnabled', false);
		updateSetting('telemetryPromptShown', true);
		setIsOpen(false);
	};

	if (loading) return null;

	return (
		<Dialog open={isOpen} onOpenChange={setIsOpen}>
			<DialogContent className="sm:max-w-[450px]">
				<DialogHeader>
					<DialogTitle>Help Improve Video Stitcher?</DialogTitle>
					<DialogDescription>
						Anonymous usage data helps us improve the app. No personal data, file names, or video content is
						collected.
					</DialogDescription>
				</DialogHeader>

				<div className="space-y-3">
					<button
						onClick={() => setShowDetails(!showDetails)}
						className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
					>
						{showDetails ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
						{showDetails ? 'Hide details' : 'Show what we collect'}
					</button>

					{showDetails && (
						<div className="space-y-2 text-sm">
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
									<li>Disable anytime in Settings</li>
								</ul>
							</div>
						</div>
					)}
				</div>

				<div className="flex items-center justify-between space-x-2 py-2">
					<Label
						htmlFor="include-system-info"
						className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
					>
						Include system info
					</Label>
					<Switch
						id="include-system-info"
						checked={includeSystemInfo}
						onCheckedChange={setIncludeSystemInfo}
					/>
				</div>

				<DialogFooter className="flex-col sm:flex-row gap-2">
					<Button variant="outline" onClick={handleDecline} className="w-full sm:w-auto">
						No Thanks
					</Button>
					<Button onClick={handleAccept} className="w-full sm:w-auto">
						Enable Telemetry
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
