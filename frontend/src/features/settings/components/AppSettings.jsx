import React from 'react';
import { useSettings } from '@/hooks/useSettings';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import { Trash2, RotateCcw, Settings2, Database, Bug } from 'lucide-react';

export default function AppSettings() {
	const { settings, updateSetting, resetSettings } = useSettings();

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

	return (
		<div className="space-y-6">
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
								Show detailed console logs for troubleshooting
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

			{/* Storage Management */}
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
							<Label className="text-base">Clear Legacy Match Flag</Label>
							<p className="text-sm text-muted-foreground">Force reload legacy matches on next launch</p>
						</div>
						<Button
							variant="outline"
							size="sm"
							onClick={() => handleClearStorage('legacyMatchesLoaded', 'Legacy match flag')}
						>
							<Trash2 className="h-4 w-4 mr-2" />
							Clear
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
								<Label className="text-base text-destructive">Clear All Data</Label>
								<Badge variant="destructive" className="text-xs">
									Destructive
								</Badge>
							</div>
							<p className="text-sm text-muted-foreground">
								Remove all localStorage data and reload application
							</p>
						</div>
						<Button variant="destructive" size="sm" onClick={() => handleClearStorage('all', 'All data')}>
							<Trash2 className="h-4 w-4 mr-2" />
							Clear All
						</Button>
					</div>
				</CardContent>
			</Card>
		</div>
	);
}
