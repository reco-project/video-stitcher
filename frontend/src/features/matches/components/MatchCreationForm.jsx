import React, { useState, useEffect } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/toast';
import { Video } from 'lucide-react';
import { getEncoderSettings } from '@/features/settings/api/settings';
import CameraSettings from './CameraSettings';
import VideoList from './VideoList';
import QualitySettings from './QualitySettings';
import { useVideoManager } from '../hooks/useVideoManager';

const DRAFT_KEY = 'matchCreationDraft';

const loadDraft = () => {
	try {
		const draft = localStorage.getItem(DRAFT_KEY);
		return draft ? JSON.parse(draft) : null;
	} catch (err) {
		console.warn('Failed to load draft:', err);
		return null;
	}
};

const saveDraft = (data) => {
	try {
		localStorage.setItem(DRAFT_KEY, JSON.stringify(data));
	} catch (err) {
		console.warn('Failed to save draft:', err);
	}
};

// TODO: Would highly benefit from more modularization and better prop management
function MatchCreationFormInner({ onSubmit, onCancel, initialData }) {
	const draft = loadDraft();
	const { showToast } = useToast();
	const [name, setName] = useState(initialData?.name || draft?.name || '');

	// Video management via hook
	const { left, right, handlers } = useVideoManager(
		initialData?.left_videos?.map((v) => v.path) || draft?.leftVideoPaths || [],
		initialData?.right_videos?.map((v) => v.path) || draft?.rightVideoPaths || []
	);

	// Profile selection
	const [leftProfileId, setLeftProfileId] = useState(draft?.leftProfileId || '');
	const [rightProfileId, setRightProfileId] = useState(draft?.rightProfileId || '');

	// Initialize profiles from initialData when editing
	useEffect(() => {
		if (initialData) {
			const leftProfileFromData =
				initialData.left_videos?.[0]?.profile_id || initialData.metadata?.left_profile_id || '';
			const rightProfileFromData =
				initialData.right_videos?.[0]?.profile_id || initialData.metadata?.right_profile_id || '';

			if (leftProfileFromData) {
				setLeftProfileId(leftProfileFromData);
			}
			if (rightProfileFromData) {
				setRightProfileId(rightProfileFromData);
			}
		}
	}, [initialData]);

	const [isSubmitting, setIsSubmitting] = useState(false);

	// Encoder info
	const [encoderInfo, setEncoderInfo] = useState(null);
	const [loadingEncoder, setLoadingEncoder] = useState(true);

	// Quality settings
	const [qualityPreset, setQualityPreset] = useState(
		initialData?.quality_settings?.preset || draft?.qualityPreset || '1080p'
	);
	const [customBitrate, setCustomBitrate] = useState(
		initialData?.quality_settings?.custom?.bitrate || draft?.customBitrate || '30M'
	);
	const [customPreset, setCustomPreset] = useState(
		initialData?.quality_settings?.custom?.preset || draft?.customPreset || 'medium'
	);
	const [customResolution, setCustomResolution] = useState(
		initialData?.quality_settings?.custom?.resolution || draft?.customResolution || '1080p'
	);
	const [customUseGpuDecode, setCustomUseGpuDecode] = useState(
		initialData?.quality_settings?.custom?.use_gpu_decode ?? draft?.customUseGpuDecode ?? true
	);

	const handleCustomSettingsChange = (changes) => {
		if ('bitrate' in changes) setCustomBitrate(changes.bitrate);
		if ('preset' in changes) setCustomPreset(changes.preset);
		if ('resolution' in changes) setCustomResolution(changes.resolution);
		if ('useGpuDecode' in changes) setCustomUseGpuDecode(changes.useGpuDecode);
	};

	// Save draft to localStorage with debounce
	useEffect(() => {
		const timeoutId = setTimeout(() => {
			const draftData = {
				name,
				leftVideoPaths: left.paths,
				rightVideoPaths: right.paths,
				leftProfileId,
				rightProfileId,
				qualityPreset,
				customBitrate,
				customPreset,
				customResolution,
				customUseGpuDecode,
			};
			saveDraft(draftData);
		}, 500);

		return () => clearTimeout(timeoutId);
	}, [
		name,
		left.paths,
		right.paths,
		leftProfileId,
		rightProfileId,
		qualityPreset,
		customBitrate,
		customPreset,
		customResolution,
		customUseGpuDecode,
	]);

	// Load encoder settings on mount
	useEffect(() => {
		getEncoderSettings()
			.then((info) => {
				setEncoderInfo(info);
			})
			.catch((err) => {
				console.error('Failed to load encoder settings:', err);
			})
			.finally(() => {
				setLoadingEncoder(false);
			});
	}, []);

	const handleSubmit = async (startProcessing = true) => {
		// Validation
		if (!name.trim()) {
			showToast({ message: 'Please enter a match name', type: 'error' });
			return;
		}

		const validLeftPaths = left.paths.filter((p) => p.trim());
		const validRightPaths = right.paths.filter((p) => p.trim());

		if (validLeftPaths.length === 0) {
			showToast({ message: 'Please select at least one left camera video', type: 'error' });
			return;
		}

		if (validRightPaths.length === 0) {
			showToast({ message: 'Please select at least one right camera video', type: 'error' });
			return;
		}

		if (!leftProfileId) {
			showToast({ message: 'Please select a lens profile for the left camera', type: 'error' });
			return;
		}

		if (!rightProfileId) {
			showToast({ message: 'Please select a lens profile for the right camera', type: 'error' });
			return;
		}

		setIsSubmitting(true);

		try {
			// Fetch profile data for submission
			const [leftProfileRes, rightProfileRes] = await Promise.all([
				fetch(`${window.BACKEND_URL || 'http://localhost:8000'}/api/profiles/${leftProfileId}`),
				fetch(`${window.BACKEND_URL || 'http://localhost:8000'}/api/profiles/${rightProfileId}`),
			]);

			if (!leftProfileRes.ok || !rightProfileRes.ok) {
				throw new Error('Failed to fetch profile data');
			}

			const leftProfile = await leftProfileRes.json();
			const rightProfile = await rightProfileRes.json();

			// Preset to bitrate mapping (frontend handles all preset logic)
			const presetToBitrate = {
				'720p': '30M',
				'1080p': '50M',
				'1440p': '70M',
			};

			// Build quality settings - always send full settings (backend has no preset logic)
			const qualitySettings =
				qualityPreset === 'custom'
					? {
							preset: 'custom',
							bitrate: customBitrate,
							speed_preset: customPreset,
							resolution: customResolution,
							use_gpu_decode: customUseGpuDecode,
						}
					: {
							preset: qualityPreset,
							bitrate: presetToBitrate[qualityPreset] || '50M',
							speed_preset: 'superfast',
							resolution: qualityPreset, // Use preset name as resolution
							use_gpu_decode: false,
						};

			await onSubmit(
				{
					name: name.trim(),
					left_videos: validLeftPaths.map((path) => ({ path, profile_id: leftProfileId })),
					right_videos: validRightPaths.map((path) => ({ path, profile_id: rightProfileId })),
					leftProfile,
					rightProfile,
					qualitySettings,
				},
				startProcessing
			);
			// Clear draft on successful submission
			try {
				localStorage.removeItem(DRAFT_KEY);
			} catch (err) {
				console.warn('Failed to clear draft:', err);
			}
		} catch (err) {
			showToast({ message: err.message || 'Failed to create match', type: 'error' });
			setIsSubmitting(false);
			throw err; // Re-throw so MatchWizard can catch it
		}
	};

	const handleCancel = () => {
		// Clear draft when canceling
		try {
			localStorage.removeItem(DRAFT_KEY);
		} catch (err) {
			console.warn('Failed to clear draft:', err);
		}
		onCancel();
	};

		return (
			<div className="w-full max-w-6xl space-y-6">
				{/* Header */}
				<div>
					<h1 className="text-3xl font-bold">Create New Match</h1>
					<p className="text-muted-foreground mt-2">
						Configure your match by selecting videos and assigning lens profiles for both cameras.
					</p>
				</div>
				{/* Match Name */}
				<Card>
					<CardHeader>
						<CardTitle>Match Name</CardTitle>
					</CardHeader>
					<CardContent>
						<Input
							type="text"
							value={name}
							onChange={(e) => setName(e.target.value)}
							placeholder="e.g., Concert 2025-12-29"
							className="text-lg"
							autoFocus={false}
						/>
					</CardContent>
				</Card>
				{/* Videos */}
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2">
								<Video className="h-5 w-5" />
								Left Camera
							</CardTitle>
						</CardHeader>
						<CardContent>
							<VideoList
								side="left"
								videoPaths={left.paths}
								metadata={left.metadata}
								onSelectFiles={handlers.handleSelectFiles}
								onRemoveVideo={handlers.handleRemoveVideo}
								onDragStart={handlers.handleDragStart}
								onDrop={handlers.handleDrop}
							/>
						</CardContent>
					</Card>
					<Card>
						<CardHeader>
							<CardTitle className="flex items-center gap-2">
								<Video className="h-5 w-5" />
								Right Camera
							</CardTitle>
						</CardHeader>
						<CardContent>
							<VideoList
								side="right"
								videoPaths={right.paths}
								metadata={right.metadata}
								onSelectFiles={handlers.handleSelectFiles}
								onRemoveVideo={handlers.handleRemoveVideo}
								onDragStart={handlers.handleDragStart}
								onDrop={handlers.handleDrop}
							/>
						</CardContent>
					</Card>
				</div>
				{/* Camera Settings */}
				<CameraSettings
					leftVideoPaths={left.paths}
					leftMetadata={left.metadata}
					leftProfileId={leftProfileId}
					onLeftProfileChange={setLeftProfileId}
					rightVideoPaths={right.paths}
					rightMetadata={right.metadata}
					rightProfileId={rightProfileId}
					onRightProfileChange={setRightProfileId}
					onSelectFiles={handlers.handleSelectFiles}
					onRemoveVideo={handlers.handleRemoveVideo}
					onDragStart={handlers.handleDragStart}
					onDrop={handlers.handleDrop}
				/>
				{/* Quality Settings */}
				<QualitySettings
					qualityPreset={qualityPreset}
					onPresetChange={setQualityPreset}
					customBitrate={customBitrate}
					customPreset={customPreset}
					customResolution={customResolution}
					customUseGpuDecode={customUseGpuDecode}
					onCustomChange={handleCustomSettingsChange}
					encoderInfo={encoderInfo}
					loadingEncoder={loadingEncoder}
				/>
				{/* Actions */}
				<div className="flex justify-between items-center pt-4">
					<Button type="button" variant="outline" onClick={handleCancel} disabled={isSubmitting}>
						Cancel
					</Button>
					{initialData ? (
						<div className="flex gap-2">
							<Button
								onClick={() => handleSubmit(false)}
								disabled={isSubmitting}
								variant="outline"
								size="lg"
								className="px-8"
							>
								{isSubmitting ? 'Saving...' : 'Save'}
							</Button>
							<Button onClick={() => handleSubmit(true)} disabled={isSubmitting} size="lg" className="px-8">
								{isSubmitting ? 'Processing...' : 'Save & Process'}
							</Button>
						</div>
					) : (
						<Button onClick={() => handleSubmit(true)} disabled={isSubmitting} size="lg" className="px-8">
							{isSubmitting ? 'Creating...' : 'Create & Process'}
						</Button>
					)}
				</div>
			</div>
		);
	}

	export default function MatchCreationForm(props) {
		return (
			<ErrorBoundary fallback={<div className="text-destructive text-sm p-2">Something went wrong creating the match.</div>}>
				<MatchCreationFormInner {...props} />
			</ErrorBoundary>
		);
	}
