import React, { useEffect, useRef, useState, useMemo } from 'react';
import { cn } from '@/lib/cn';
import { useViewerStore } from '../stores/store';
import { 
	LucidePlay, 
	LucidePause, 
	LucideVolume, 
	LucideMaximize, 
	LucideCircle, 
	LucideSquare,
	LucideMic,
	LucideMicOff,
	LucideSettings,
	LucideChevronDown
} from 'lucide-react';
import { useCanvasRecorder } from '../hooks/useCanvasRecorder';
import { useSettings } from '@/hooks/useSettings';
import { useNavigate } from 'react-router-dom';
import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu';
import { DropdownMenuItem } from '@/components/ui/dropdown-menu';

export default function VideoPlayer({ children, className }) {
	const videoRef = useViewerStore((state) => state.videoRef);
	const containerRef = useRef(null);
	const [playing, setPlaying] = useState(false);
	const [muted, setMuted] = useState(false);
	const [volume, setVolume] = useState(1);
	const [currentTime, setCurrentTime] = useState(0);
	const [isFullscreen, setIsFullscreen] = useState(false);
	const [micEnabled, setMicEnabled] = useState(false);
	const [micStream, setMicStream] = useState(null);
	const [audioDevices, setAudioDevices] = useState([]);
	const [selectedDeviceId, setSelectedDeviceId] = useState(null);
	const navigate = useNavigate();

	// Get recording settings
	const { settings } = useSettings();
	const recordingOptions = useMemo(() => ({
		fps: 30, // Fixed at 30 FPS for now
		videoBitsPerSecond: (settings.recordingBitrate ?? 16) * 1000000,
		mimeType: settings.recordingFormat === 'webm-vp8' 
			? 'video/webm;codecs=vp8' 
			: 'video/webm;codecs=vp9',
	}), [settings.recordingBitrate, settings.recordingFormat]);

	// Canvas recording
	const { isRecording, recordingDuration, toggleRecording } = useCanvasRecorder(recordingOptions);

	// Enumerate audio devices
	const enumerateAudioDevices = async () => {
		try {
			const devices = await navigator.mediaDevices.enumerateDevices();
			const audioInputs = devices.filter(device => device.kind === 'audioinput');
			setAudioDevices(audioInputs);
			if (audioInputs.length > 0 && !selectedDeviceId) {
				setSelectedDeviceId(audioInputs[0].deviceId);
			}
		} catch (err) {
			console.error('Failed to enumerate audio devices:', err);
		}
	};

	// Toggle microphone
	const handleToggleMic = async () => {
		if (micEnabled && micStream) {
			// Stop microphone
			micStream.getTracks().forEach(track => track.stop());
			setMicStream(null);
			setMicEnabled(false);
		} else {
			// Request microphone access
			try {
				const constraints = {
					audio: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true
				};
				const stream = await navigator.mediaDevices.getUserMedia(constraints);
				setMicStream(stream);
				setMicEnabled(true);
				// Enumerate devices after first access to get labels
				await enumerateAudioDevices();
			} catch (err) {
				console.error('Failed to access microphone:', err);
				alert('Could not access microphone. Please check your browser permissions.');
			}
		}
	};

	// Change microphone device
	const handleChangeMicrophone = async (deviceId) => {
		setSelectedDeviceId(deviceId);
		if (micEnabled && micStream) {
			// Stop current stream
			micStream.getTracks().forEach(track => track.stop());
			// Start with new device
			try {
				const stream = await navigator.mediaDevices.getUserMedia({
					audio: { deviceId: { exact: deviceId } }
				});
				setMicStream(stream);
			} catch (err) {
				console.error('Failed to switch microphone:', err);
				setMicEnabled(false);
			}
		}
	};

	// Cleanup mic stream on unmount
	useEffect(() => {
		return () => {
			if (micStream) {
				micStream.getTracks().forEach(track => track.stop());
			}
		};
	}, [micStream]);

	const handleToggleRecording = () => {
		// If currently recording, stop and pause the video
		if (isRecording) {
			const canvas = containerRef.current?.querySelector('canvas');
			if (canvas) {
				toggleRecording(canvas, videoRef, micStream);
			}
			// Pause the video when stopping recording
			if (videoRef && !videoRef.paused) {
				videoRef.pause();
				setPlaying(false);
			}
			return;
		}

		// Suggest fullscreen for better resolution
		if (!document.fullscreenElement) {
			const shouldContinue = window.confirm(
				'For best recording quality, fullscreen mode is recommended.\n\n' +
				'Recording in fullscreen captures at your screen\'s native resolution.\n\n' +
				'Click OK to start recording anyway, or Cancel to go fullscreen first.'
			);
			if (!shouldContinue) {
				// Try to go fullscreen
				const el = containerRef.current;
				if (el?.requestFullscreen) {
					el.requestFullscreen();
				}
				return;
			}
		}

		// Find the canvas element inside the container
		const canvas = containerRef.current?.querySelector('canvas');
		if (canvas) {
			// Auto-play video if not playing when starting recording
			if (videoRef && videoRef.paused) {
				videoRef.play();
				setPlaying(true);
			}
			toggleRecording(canvas, videoRef, micStream);
		} else {
			console.warn('Canvas not found for recording');
		}
	};

	const formatRecordingTime = (seconds) => {
		const mins = Math.floor(seconds / 60);
		const secs = seconds % 60;
		return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
	};

	const progress = videoRef && videoRef.duration ? (currentTime / videoRef.duration) * 100 : 0;

	useEffect(() => {
		if (!videoRef) return;
		let interval = null;
		if (playing) {
			// maybe tying interval to videoRef changes only (without considering playing) is sufficient...
			interval = setInterval(() => {
				if (videoRef) {
					setCurrentTime(videoRef.currentTime);
				}
			}, 1000);
		}
		return () => {
			if (interval) clearInterval(interval);
		};
	}, [videoRef, playing]);

	// Sync playing state with video element. Keep it, even if we have interval above. It provides more immediate feedback.
	useEffect(() => {
		if (!videoRef) return;
		const onTimeUpdate = () => setPlaying(!videoRef.paused);
		videoRef.addEventListener('timeupdate', onTimeUpdate);
		return () => {
			if (videoRef) {
				videoRef.removeEventListener('timeupdate', onTimeUpdate);
			}
		};
	}, [videoRef, setPlaying]);

	// Sync fullscreen state with document fullscreen changes (e.g., when user presses Escape)
	useEffect(() => {
		const onFullscreenChange = () => {
			setIsFullscreen(!!document.fullscreenElement);
		};
		document.addEventListener('fullscreenchange', onFullscreenChange);
		return () => document.removeEventListener('fullscreenchange', onFullscreenChange);
	}, []);

	// Enumerate audio devices on mount
	useEffect(() => {
		enumerateAudioDevices();
	}, []);

	// Global keyboard shortcuts for the viewer
	useEffect(() => {
		const handleKeyDown = (e) => {
			// Don't trigger shortcuts when typing in input fields or interacting with sliders
			const target = e.target;
			if (
				target.tagName === 'INPUT' ||
				target.tagName === 'TEXTAREA' ||
				target.isContentEditable ||
				target.getAttribute('role') === 'slider'
			) {
				return;
			}

			switch (e.key.toLowerCase()) {
				case ' ':
				case 'k':
					e.preventDefault();
					if (videoRef) {
						if (videoRef.paused) {
							videoRef.play();
							setPlaying(true);
						} else {
							videoRef.pause();
							setPlaying(false);
						}
					}
					break;
				case 'arrowright':
				case 'l':
					if (videoRef) {
						videoRef.currentTime = Math.min(videoRef.currentTime + 5, videoRef.duration);
						setCurrentTime(videoRef.currentTime);
					}
					break;
				case 'arrowleft':
				case 'j':
					if (videoRef) {
						videoRef.currentTime = Math.max(videoRef.currentTime - 5, 0);
						setCurrentTime(videoRef.currentTime);
					}
					break;
				case 'm':
					if (videoRef) {
						videoRef.muted = !videoRef.muted;
						setMuted(videoRef.muted);
					}
					break;
				case 'f':
					handleFullscreenRef.current();
					break;
				case 'arrowup':
					e.preventDefault();
					if (videoRef) {
						const newVolume = Math.min(videoRef.volume + 0.1, 1);
						videoRef.volume = newVolume;
						setVolume(newVolume);
					}
					break;
				case 'arrowdown':
					e.preventDefault();
					if (videoRef) {
						const newVolume = Math.max(videoRef.volume - 0.1, 0);
						videoRef.volume = newVolume;
						setVolume(newVolume);
					}
					break;
				case 'r':
					handleToggleRecordingRef.current();
					break;
				default:
					break;
			}
		};

		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [videoRef]);

	const togglePlay = () => {
		if (!videoRef) return;
		if (videoRef.paused) {
			videoRef.play();
			setPlaying(true);
		} else {
			videoRef.pause();
			setPlaying(false);
		}
	};

	const handleVolumeToggle = () => {
		if (!videoRef) return;
		videoRef.muted = !videoRef.muted;
		setMuted(videoRef.muted);
	};

	const handleVolumeChange = (e) => {
		if (!videoRef) return;
		videoRef.volume = parseFloat(e.target.value);
		setVolume(videoRef.volume);
	};

	const formatTime = (time) => {
		if (isNaN(time)) return '00:00';
		const minutes = Math.floor(time / 60);
		const seconds = Math.floor(time % 60);
		return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
	};

	const handleSeek = (e) => {
		if (!videoRef || !videoRef.duration) return;
		const rect = e.currentTarget.getBoundingClientRect();
		const clickX = e.clientX - rect.left;
		const seekPercent = Math.max(0, Math.min(1, clickX / rect.width));
		videoRef.currentTime = seekPercent * videoRef.duration;
		setCurrentTime(videoRef.currentTime);
	};

	const handleMouseDown = (e) => {
		if (!videoRef || !videoRef.duration) return;
		const video = videoRef; // Capture reference
		const rect = e.currentTarget.getBoundingClientRect();
		const startX = e.clientX;
		const seek = (clientX) => {
			if (!video) return;
			const clickX = clientX - rect.left;
			const seekPercent = Math.max(0, Math.min(1, clickX / rect.width));
			video.currentTime = seekPercent * video.duration;
			setCurrentTime(video.currentTime);
		};
		seek(startX);

		const onMouseMove = (moveEvent) => {
			seek(moveEvent.clientX);
		};
		const onMouseUp = (upEvent) => {
			seek(upEvent.clientX);
			window.removeEventListener('mousemove', onMouseMove);
			window.removeEventListener('mouseup', onMouseUp);
		};
		window.addEventListener('mousemove', onMouseMove);
		window.addEventListener('mouseup', onMouseUp);
	};

	const handleFullscreen = async () => {
		const el = containerRef.current;
		if (!el) return;

		try {
			if (document.fullscreenElement) {
				// Warn if trying to exit while recording
				if (isRecording) {
					const shouldExit = window.confirm(
						'You are currently recording.\n\n' +
						'Exiting fullscreen will reduce the recording resolution.\n\n' +
						'Click OK to exit fullscreen, or Cancel to stay in fullscreen.'
					);
					if (!shouldExit) return;
				}
				await document.exitFullscreen();
			} else if (el.requestFullscreen) {
				await el.requestFullscreen();
			} else if (el.webkitRequestFullscreen) {
				el.webkitRequestFullscreen();
			} else if (el.msRequestFullscreen) {
				el.msRequestFullscreen();
			} else {
				console.warn('Fullscreen API is not supported.');
			}
		} catch (err) {
			// Ignore errors from rapid fullscreen toggling or inactive document
			console.warn('Fullscreen operation failed:', err.message);
		}
	};

	// Ref to access handleFullscreen in the global event listener
	const handleFullscreenRef = useRef(handleFullscreen);
	useEffect(() => {
		handleFullscreenRef.current = handleFullscreen;
	}, [handleFullscreen]);

	// Ref to access handleToggleRecording in the global event listener
	const handleToggleRecordingRef = useRef(handleToggleRecording);
	useEffect(() => {
		handleToggleRecordingRef.current = handleToggleRecording;
	}, [handleToggleRecording]);

	const [isControlsVisible, setIsControlsVisible] = useState(true);

	const handleMouseEnter = () => setIsControlsVisible(true);
	const handleMouseLeave = () => setIsControlsVisible(false); // TODO: refine better logic for hiding controls after timeout

	return (
		<div
			className={cn(
				`relative w-full max-w-5xl bg-black overflow-hidden ${isFullscreen ? 'rounded-none' : 'rounded-lg'}`,
				className
			)}
			ref={containerRef}
			onMouseEnter={handleMouseEnter}
			onMouseLeave={handleMouseLeave}
		>
			<div className="w-full h-full aspect-video">{children}</div>

			{/* Controls Overlay */}
			{isControlsVisible && (
				<div className="absolute bottom-0 left-0 w-full bg-black/60 backdrop-blur-sm text-white px-4 py-2 flex items-center space-x-4 z-10"
					onPointerDown={(e) => e.stopPropagation()}
					onPointerUp={(e) => e.stopPropagation()}
					onPointerMove={(e) => e.stopPropagation()}
				>
					{/* Play/Pause */}
					<button
						onClick={togglePlay}
						className="p-1 hover:text-primary transition cursor-pointer"
						aria-label={playing ? 'Pause' : 'Play'}
					>
						{playing ? <LucidePause /> : <LucidePlay />}
					</button>
					{/* Progress Bar with Drag Seek */}
					<div
						className="flex-1 h-2 bg-white/30 rounded cursor-pointer relative"
						onClick={handleSeek}
						onMouseDown={handleMouseDown}
						role="slider"
						aria-valuemin="0"
						aria-valuemax="100"
						aria-valuenow={progress}
						tabIndex={0}
						onKeyDown={(e) => {
							if (e.key === 'ArrowRight') {
								handleSeek({
									currentTarget: e.currentTarget,
									clientX: e.clientX + 30,
								});
							}
							if (e.key === 'ArrowLeft') {
								handleSeek({
									currentTarget: e.currentTarget,
									clientX: e.clientX - 30,
								});
							}
						}}
					>
						<div className="h-2 bg-primary rounded pointer-events-none" style={{ width: `${progress}%` }} />
						{/* Cursor/Thumb */}
						<div className="absolute top-1/2 -translate-y-1/2" style={{ left: `calc(${progress}% - 8px)` }}>
							<div className="w-4 h-4 bg-primary rounded-full pointer-events-none border-2 border-white shadow" />
						</div>
					</div>
					{/* Time Display */}
					<div className="flex items-center space-x-2">
						<span>{formatTime(currentTime)}</span>
						<span>/</span>
						<span>{videoRef ? formatTime(videoRef.duration) : '00:00'}</span>
					</div>
					<div className="flex items-center space-x-2">
						<LucideVolume
							onClick={handleVolumeToggle}
							className="cursor-pointer"
							strokeWidth={muted ? 0 : 2}
							aria-label={muted ? 'Unmute' : 'Mute'}
						/>
						<input
							type="range"
							min="0"
							max="1"
							step="0.01"
							value={muted ? 0 : volume}
							onChange={handleVolumeChange}
							className="w-24 cursor-pointer"
							aria-label="Volume control"
						/>
					</div>
					{/* Record */}
					<div className="flex items-center gap-1 border-l border-white/20 pl-3">
						{/* Mic Toggle with Device Selection */}
						{isRecording ? (
							<button
								disabled
								className={cn(
									"p-1.5 rounded transition flex items-center gap-0.5 cursor-not-allowed opacity-50",
									micEnabled 
										? "text-green-500 bg-green-500/20" 
										: "text-white/60"
								)}
								aria-label="Microphone locked during recording"
								title="Cannot change microphone while recording"
							>
								{micEnabled ? <LucideMic className="h-4 w-4" /> : <LucideMicOff className="h-4 w-4" />}
								<LucideChevronDown className="h-3 w-3" />
							</button>
						) : (
							<DropdownMenuPrimitive.Root modal={false}>
								<DropdownMenuPrimitive.Trigger asChild>
									<button
										className={cn(
											"p-1.5 rounded transition cursor-pointer flex items-center gap-0.5",
											micEnabled 
												? "text-green-500 hover:text-green-400 bg-green-500/20" 
												: "text-white/60 hover:text-white"
										)}
										aria-label={micEnabled ? 'Microphone Settings' : 'Enable Microphone'}
										title={micEnabled ? 'Microphone On (click to configure)' : 'Configure Microphone'}
									>
										{micEnabled ? <LucideMic className="h-4 w-4" /> : <LucideMicOff className="h-4 w-4" />}
										<LucideChevronDown className="h-3 w-3" />
									</button>
								</DropdownMenuPrimitive.Trigger>
								<DropdownMenuPrimitive.Portal container={containerRef.current}>
									<DropdownMenuPrimitive.Content 
										align="end"
										sideOffset={5}
										className="z-[9999] w-56 min-w-32 overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2"
									>
								<DropdownMenuItem
									onClick={handleToggleMic}
									className="cursor-pointer"
								>
									{micEnabled ? (
										<>
											<LucideMicOff className="h-4 w-4 mr-2" />
											Disable Microphone
										</>
									) : (
										<>
											<LucideMic className="h-4 w-4 mr-2" />
											Enable Microphone
										</>
									)}
								</DropdownMenuItem>
								{audioDevices.length > 0 && (
									<>
										<div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">
											Select Device
										</div>
										{audioDevices.map(device => (
											<DropdownMenuItem
												key={device.deviceId}
												onClick={() => handleChangeMicrophone(device.deviceId)}
												className={cn(
													"cursor-pointer",
													selectedDeviceId === device.deviceId && "bg-accent"
												)}
											>
												{device.label || `Microphone ${device.deviceId.slice(0, 8)}`}
											</DropdownMenuItem>
										))}
									</>
								)}
									</DropdownMenuPrimitive.Content>
								</DropdownMenuPrimitive.Portal>
							</DropdownMenuPrimitive.Root>
						)}

						{/* Record Button */}
						<button
							onClick={handleToggleRecording}
							className={cn(
								"p-1.5 rounded transition cursor-pointer flex items-center gap-1",
								isRecording 
									? "bg-red-600 hover:bg-red-500 text-white" 
									: "bg-red-600/80 hover:bg-red-500 text-white"
							)}
							aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
							title={isRecording ? 'Stop Recording (R)' : 'Start Recording (R)'}
						>
							{isRecording ? (
								<>
									<LucideSquare className="h-3.5 w-3.5 fill-current" />
									<span className="text-xs font-mono">{formatRecordingTime(recordingDuration)}</span>
								</>
							) : (
								<LucideCircle className="h-3.5 w-3.5 fill-current" />
							)}
						</button>

						{/* Settings Button */}
						<button
							onClick={() => navigate('/profiles?tab=settings#recording')}
							className="p-1.5 text-white/60 hover:text-white transition cursor-pointer"
							aria-label="Recording Settings"
							title="Recording Settings"
						>
							<LucideSettings className="h-4 w-4" />
						</button>
					</div>

					{/* Fullscreen */}
					<button
						onClick={handleFullscreen}
						className="p-1 hover:text-primary transition cursor-pointer"
						aria-label="Toggle Fullscreen"
					>
						<LucideMaximize />
					</button>
				</div>
			)}
		</div>
	);
}
