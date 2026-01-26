import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useToast } from '@/components/ui/toast';
import { useSettings } from '@/hooks/useSettings';
import CameraSettings from './CameraSettings';
import VideoSections from './VideoSections';
import QualitySettings from './QualitySettings';
import { useMatchDraft, useAutoSaveDraft } from '../hooks/useMatchDraft';
import { useQualitySettings } from '../hooks/useQualitySettings';

function MatchCreationFormInner({ onSubmit, onCancel, initialData }) {
	const { settings } = useSettings();
	const { loadDraft, saveDraft, clearDraft } = useMatchDraft();
	const draft = loadDraft();
	const { showToast } = useToast();
	
	const [name, setName] = useState(initialData?.name || draft?.name || '');
	const [isSubmitting, setIsSubmitting] = useState(false);

	// Video data from VideoSections
	const [videoData, setVideoData] = useState({
		left: { paths: initialData?.left_videos?.map((v) => v.path) || draft?.leftVideoPaths || [], metadata: [] },
		right: { paths: initialData?.right_videos?.map((v) => v.path) || draft?.rightVideoPaths || [], metadata: [] },
	});

	const handleVideoChange = useCallback(({ left, right }) => {
		setVideoData({ left, right });
	}, []);

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

			if (leftProfileFromData) setLeftProfileId(leftProfileFromData);
			if (rightProfileFromData) setRightProfileId(rightProfileFromData);
		}
	}, [initialData]);

	// Quality settings (uses dedicated hook)
	const quality = useQualitySettings({
		preset: initialData?.quality_settings?.preset || draft?.qualityPreset,
		customBitrate: initialData?.quality_settings?.custom?.bitrate || draft?.customBitrate,
		customPreset: initialData?.quality_settings?.custom?.preset || draft?.customPreset,
		customResolution: initialData?.quality_settings?.custom?.resolution || draft?.customResolution,
		customUseGpuDecode: initialData?.quality_settings?.custom?.use_gpu_decode ?? draft?.customUseGpuDecode,
	});

	// Auto-save draft
	const draftData = useMemo(
		() => ({
			name,
			leftVideoPaths: videoData.left.paths,
			rightVideoPaths: videoData.right.paths,
			leftProfileId,
			rightProfileId,
			...quality.draftValues,
		}),
		[name, videoData.left.paths, videoData.right.paths, leftProfileId, rightProfileId, quality.draftValues]
	);
	useAutoSaveDraft(draftData, saveDraft);

	const handleSubmit = async (startProcessing = true) => {
		// Validation
		if (!name.trim()) {
			showToast({ message: 'Please enter a match name', type: 'error' });
			return;
		}

		const validLeftPaths = videoData.left.paths.filter((p) => p.trim());
		const validRightPaths = videoData.right.paths.filter((p) => p.trim());

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

			await onSubmit(
				{
					name: name.trim(),
					left_videos: validLeftPaths.map((path) => ({ path, profile_id: leftProfileId })),
					right_videos: validRightPaths.map((path) => ({ path, profile_id: rightProfileId })),
					leftProfile,
					rightProfile,
					qualitySettings: quality.qualitySettings,
				},
				startProcessing
			);
			clearDraft();
		} catch (err) {
			showToast({ message: err.message || 'Failed to create match', type: 'error' });
			setIsSubmitting(false);
			throw err; // Re-throw so MatchWizard can catch it
		}
	};

	const handleCancel = () => {
		clearDraft();
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
							placeholder="e.g., My game 2025-12-29"
							className="text-lg"
							autoFocus={false}
						/>
					</CardContent>
				</Card>
				{/* Videos */}
				<VideoSections
					initialLeftPaths={initialData?.left_videos?.map((v) => v.path) || draft?.leftVideoPaths || []}
					initialRightPaths={initialData?.right_videos?.map((v) => v.path) || draft?.rightVideoPaths || []}
					onChange={handleVideoChange}
				/>
				{/* Camera Settings */}
				<CameraSettings
					leftProfileId={leftProfileId}
					onLeftProfileChange={setLeftProfileId}
					rightProfileId={rightProfileId}
					onRightProfileChange={setRightProfileId}
				/>
				{/* Quality Settings */}
				<QualitySettings
					qualityPreset={quality.preset}
					onPresetChange={quality.setPreset}
					customBitrate={quality.customBitrate}
					customPreset={quality.customPreset}
					customResolution={quality.customResolution}
					customUseGpuDecode={quality.customUseGpuDecode}
					onCustomChange={quality.handleCustomChange}
					encoderInfo={quality.encoderInfo}
					loadingEncoder={quality.loadingEncoder}
				/>
				{/* Debug panel - only shown when debugMode is enabled in settings */}
				{settings.debugMode && (
					<details className="text-xs bg-muted p-3 rounded-md">
						<summary className="cursor-pointer font-medium">Debug: Form State</summary>
						<pre className="mt-2 overflow-auto max-h-64 text-[10px]">
							{JSON.stringify(
								{
									name,
									leftVideoPaths: videoData.left.paths,
									rightVideoPaths: videoData.right.paths,
									leftProfileId,
									rightProfileId,
									qualitySettings: quality.qualitySettings,
									encoderInfo: quality.encoderInfo,
								},
								null,
								2
							)}
						</pre>
					</details>
				)}
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
