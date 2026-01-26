import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Camera, ExternalLink } from 'lucide-react';
import { Link } from 'react-router-dom';
import ProfileCombobox from './ProfileCombobox';

export default function CameraSettings({
	// Left camera props
	leftProfileId,
	onLeftProfileChange,
	// Right camera props
	rightProfileId,
	onRightProfileChange,
}) {
	const [sameCameraSettings, setSameCameraSettings] = useState(() => {
		const cached = localStorage.getItem('sameCameraSettings');
		return cached !== null ? JSON.parse(cached) : true;
	});

	useEffect(() => {
		localStorage.setItem('sameCameraSettings', JSON.stringify(sameCameraSettings));
	}, [sameCameraSettings]);

	const handleSameCameraToggle = (value) => {
		setSameCameraSettings(value);
		// Clear selections when switching modes
		onLeftProfileChange(null);
		onRightProfileChange(null);
	};

	const handleLeftProfileChange = (profileId) => {
		onLeftProfileChange(profileId);
		if (sameCameraSettings) {
			// Auto-sync to right when in "Same" mode
			onRightProfileChange(profileId);
		}
	};

	return (
		<Card>
			<CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
				<CardTitle className="flex items-center gap-2">
					<Camera className="h-5 w-5" />
					Camera Settings
				</CardTitle>
				<div className="flex items-center gap-3">
					<Link
						to="/profiles"
						className="text-xs text-primary hover:underline flex items-center gap-1 whitespace-nowrap shrink-0"
					>
						Manage profiles
						<ExternalLink className="h-3 w-3" />
					</Link>
					<div className="inline-flex rounded-lg border p-0.5 gap-0.5">
						<button
							type="button"
							onClick={() => handleSameCameraToggle(true)}
							title="Both videos use the same lens profile"
							className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
								sameCameraSettings ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
							}`}
						>
							Same lens profile
						</button>
						<button
							type="button"
							onClick={() => handleSameCameraToggle(false)}
							title="Videos recorded with different FOV settings"
							className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
								!sameCameraSettings ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'
							}`}
						>
							Different lens profiles
						</button>
					</div>
				</div>
			</CardHeader>
			<CardContent>
				{/* Profile Selection */}
				{sameCameraSettings ? (
					<div>
						<ProfileCombobox value={leftProfileId} onChange={handleLeftProfileChange} className="w-full" />
					</div>
				) : (
					<div className="flex gap-3 min-w-0">
						<div className="flex-1 min-w-0 rounded-lg">
							<ProfileCombobox
								value={leftProfileId}
								onChange={onLeftProfileChange}
								className="w-full"
								labelPrefix="Left: "
							/>
						</div>
						<div className="flex-1 min-w-0 rounded-lg">
							<ProfileCombobox
								value={rightProfileId}
								onChange={onRightProfileChange}
								className="w-full"
								labelPrefix="Right: "
							/>
						</div>
					</div>
				)}
			</CardContent>
		</Card>
	);
}
