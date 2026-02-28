import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import * as THREE from 'three';
import { Canvas, useThree } from '@react-three/fiber';
import { ErrorBoundary } from 'react-error-boundary';
import dewarpShader from './shaders/dewarpShader.js';
import profiles from './profiles.js';
import { useCustomVideoTexture } from '../viewer/hooks/useCustomVideoTexture.js';
import { useViewerStore } from '../viewer/stores/store.js';
import { CameraProvider, useCameraControls } from '../viewer/stores/cameraContext.jsx';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { LucidePlay, LucidePause, LucideMaximize, LucideUpload } from 'lucide-react';

// Pre-compute shader strings (pure function, always returns the same result)
const shaderConfig = dewarpShader();

/**
 * Format uniforms for the dewarping shader (full-image or split-half mode).
 * In split-half mode, each half gets its own intrinsics and distortion coefficients.
 */
function formatSplitUniforms(profile, texture) {
	const { width, height, splitPoint, blendWidth, splitHalf, left, right } = profile;
	const isSplit = splitHalf !== false;

	// In full-image mode, normalize to full image dimensions
	// In split-half mode, normalize to per-half dimensions
	const lw = isSplit ? width * splitPoint : width;
	const lh = height;
	const rw = isSplit ? width * (1 - splitPoint) : width;
	const rh = height;

	return {
		uVideo: { value: texture },
		splitPoint: { value: splitPoint },
		blendWidth: { value: blendWidth },
		splitHalf: { value: isSplit ? 1.0 : 0.0 },
		// Left/full-image lens (normalized)
		lFx: { value: left.fx / lw },
		lFy: { value: left.fy / lh },
		lCx: { value: left.cx / lw },
		lCy: { value: left.cy / lh },
		lD: { value: new THREE.Vector4(...left.d) },
		// Right lens (normalized, only used in split mode)
		rFx: { value: right.fx / rw },
		rFy: { value: right.fy / rh },
		rCx: { value: right.cx / rw },
		rCy: { value: right.cy / rh },
		rD: { value: new THREE.Vector4(...right.d) },
	};
}

/**
 * Pan/zoom controls. Camera faces -Z, looking at a flat plane.
 */
const PanoramaControls = () => {
	const { camera, gl } = useThree();
	const { yawRange, pitchRange } = useCameraControls();
	const dragging = useRef(false);
	const lastX = useRef(0);
	const lastY = useRef(0);

	const panSensitivity = 0.005;
	const zoomSensitivity = 0.05;
	const minFov = 15;
	const maxFov = 90;

	const yawQuat = useRef(new THREE.Quaternion());
	const pitchQuat = useRef(new THREE.Quaternion());
	const tempVec = useRef(new THREE.Vector3());
	const tempEuler = useRef(new THREE.Euler());

	const pitchRangeRad = THREE.MathUtils.degToRad(pitchRange);
	const yawRangeRad = THREE.MathUtils.degToRad(yawRange);

	const minPitch = -pitchRangeRad / 2;
	const maxPitch = pitchRangeRad / 2;
	const minYaw = -yawRangeRad / 2;
	const maxYaw = yawRangeRad / 2;

	useEffect(() => {
		const canvas = gl?.domElement;
		if (!canvas) return;

		const onPointerDown = (e) => {
			dragging.current = true;
			lastX.current = e.clientX;
			lastY.current = e.clientY;
			canvas.style.cursor = 'grabbing';
		};

		const onPointerUp = () => {
			dragging.current = false;
			canvas.style.cursor = 'grab';
		};

		const onPointerMove = (e) => {
			if (!dragging.current) return;
			const deltaX = e.clientX - lastX.current;
			const deltaY = e.clientY - lastY.current;
			lastX.current = e.clientX;
			lastY.current = e.clientY;

			yawQuat.current.setFromAxisAngle(tempVec.current.set(0, 1, 0), deltaX * panSensitivity);
			camera.quaternion.premultiply(yawQuat.current);

			tempVec.current.set(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
			pitchQuat.current.setFromAxisAngle(tempVec.current, deltaY * panSensitivity);
			camera.quaternion.premultiply(pitchQuat.current);

			const euler = tempEuler.current.setFromQuaternion(camera.quaternion, 'YXZ');
			euler.x = THREE.MathUtils.clamp(euler.x, minPitch, maxPitch);
			euler.y = THREE.MathUtils.clamp(euler.y, minYaw, maxYaw);
			camera.quaternion.setFromEuler(euler);
		};

		const onWheel = (e) => {
			e.preventDefault();
			camera.fov = THREE.MathUtils.clamp(camera.fov + e.deltaY * zoomSensitivity, minFov, maxFov);
			camera.updateProjectionMatrix();
		};

		canvas.style.cursor = 'grab';
		canvas.addEventListener('wheel', onWheel, { passive: false });
		canvas.addEventListener('pointerdown', onPointerDown);
		canvas.addEventListener('pointerup', onPointerUp);
		canvas.addEventListener('pointermove', onPointerMove);
		canvas.addEventListener('pointerleave', onPointerUp);

		return () => {
			canvas.style.cursor = '';
			canvas.removeEventListener('wheel', onWheel);
			canvas.removeEventListener('pointerdown', onPointerDown);
			canvas.removeEventListener('pointerup', onPointerUp);
			canvas.removeEventListener('pointermove', onPointerMove);
			canvas.removeEventListener('pointerleave', onPointerUp);
		};
	}, [camera, gl, yawRange, pitchRange]);

	return null;
};

/**
 * Create a half-cylinder geometry for 180-degree panoramic projection.
 * Maps the panoramic texture onto the inside of a half-cylinder.
 * The perspective camera at the origin looking at this surface produces
 * a mathematically correct cylindrical-to-rectilinear reprojection,
 * making straight lines in the real world appear straight in the viewport.
 */
function createHalfCylinderGeometry(radius, height, widthSegments, heightSegments) {
	const positions = [];
	const uvs = [];
	const indices = [];

	for (let iy = 0; iy <= heightSegments; iy++) {
		const v = iy / heightSegments;
		const y = (v - 0.5) * height;

		for (let ix = 0; ix <= widthSegments; ix++) {
			const u = ix / widthSegments;
			// Map u to angle: 0 -> +PI/2, 1 -> -PI/2
			const theta = (0.5 - u) * Math.PI;

			positions.push(
				radius * Math.sin(theta), // x
				y, // y
				-radius * Math.cos(theta) // z (negative = in front of camera)
			);
			uvs.push(1 - u, v);
		}
	}

	for (let iy = 0; iy < heightSegments; iy++) {
		for (let ix = 0; ix < widthSegments; ix++) {
			const a = iy * (widthSegments + 1) + ix;
			const b = a + 1;
			const c = a + (widthSegments + 1);
			const d = c + 1;
			indices.push(a, b, c);
			indices.push(b, d, c);
		}
	}

	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
	geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
	geo.setIndex(indices);
	geo.computeVertexNormals();
	return geo;
}

/**
 * Panoramic video surface using half-cylinder geometry.
 * The cylindrical geometry handles the primary projection correction.
 * The shader can still apply residual barrel correction if needed.
 */
const PanoramaPlane = ({ texture, profile }) => {
	const uniforms = useMemo(() => formatSplitUniforms(profile, texture), [profile, texture]);
	const aspect = profile.width / profile.height;

	// Half-cylinder: radius=1, height proportional to vertical FOV
	// For 180-degree horizontal, the vertical extent is height/width * PI * radius
	const radius = 1;
	const cylinderHeight = (1 / aspect) * Math.PI * radius;

	const geometry = useMemo(
		() => createHalfCylinderGeometry(radius, cylinderHeight, 128, 32),
		[radius, cylinderHeight]
	);

	return (
		<mesh geometry={geometry}>
			<shaderMaterial uniforms={uniforms} {...shaderConfig} side={THREE.BackSide} />
		</mesh>
	);
};

/**
 * Video scene: loads the video texture and renders the dewarped plane.
 */
const PanoramaScene = ({ videoSrc, profile }) => {
	const texture = useCustomVideoTexture(videoSrc);
	if (!texture) return null;
	return <PanoramaPlane texture={texture} profile={profile} />;
};

/**
 * Camera range wrapper.
 */
const CameraControlsWrapper = ({ yawRange, pitchRange, children }) => {
	const { setYawRange, setPitchRange } = useCameraControls();
	useEffect(() => {
		setYawRange(yawRange);
		setPitchRange(pitchRange);
	}, [yawRange, pitchRange, setYawRange, setPitchRange]);
	return <>{children}</>;
};

/**
 * Simple video player controls overlay.
 */
const PlayerControls = ({ containerRef }) => {
	const videoRef = useViewerStore((s) => s.videoRef);
	const [playing, setPlaying] = useState(false);
	const [currentTime, setCurrentTime] = useState(0);

	useEffect(() => {
		if (!videoRef) return;
		const interval = setInterval(() => {
			if (videoRef) setCurrentTime(videoRef.currentTime);
		}, 500);
		return () => clearInterval(interval);
	}, [videoRef]);

	useEffect(() => {
		if (!videoRef) return;
		const onPlay = () => setPlaying(true);
		const onPause = () => setPlaying(false);
		videoRef.addEventListener('play', onPlay);
		videoRef.addEventListener('pause', onPause);
		return () => {
			videoRef.removeEventListener('play', onPlay);
			videoRef.removeEventListener('pause', onPause);
		};
	}, [videoRef]);

	const handleFullscreen = useCallback(async () => {
		const el = containerRef?.current;
		if (!el) return;
		if (document.fullscreenElement) await document.exitFullscreen();
		else if (el.requestFullscreen) await el.requestFullscreen();
	}, []);

	useEffect(() => {
		const handleKeyDown = (e) => {
			if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
			switch (e.key.toLowerCase()) {
				case ' ':
				case 'k':
					e.preventDefault();
					if (videoRef) {
						if (videoRef.paused) videoRef.play();
						else videoRef.pause();
					}
					break;
				case 'arrowright':
				case 'l':
					if (videoRef) videoRef.currentTime = Math.min(videoRef.currentTime + 5, videoRef.duration);
					break;
				case 'arrowleft':
				case 'j':
					if (videoRef) videoRef.currentTime = Math.max(videoRef.currentTime - 5, 0);
					break;
				case 'f':
					handleFullscreen();
					break;
			}
		};
		window.addEventListener('keydown', handleKeyDown);
		return () => window.removeEventListener('keydown', handleKeyDown);
	}, [videoRef, handleFullscreen]);

	const togglePlay = () => {
		if (!videoRef) return;
		if (videoRef.paused) videoRef.play();
		else videoRef.pause();
	};

	const handleSeek = (e) => {
		if (!videoRef || !videoRef.duration) return;
		const rect = e.currentTarget.getBoundingClientRect();
		const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
		videoRef.currentTime = pct * videoRef.duration;
		setCurrentTime(videoRef.currentTime);
	};

	const formatTime = (t) => {
		if (isNaN(t)) return '00:00';
		const m = Math.floor(t / 60);
		const s = Math.floor(t % 60);
		return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
	};

	const progress = videoRef && videoRef.duration ? (currentTime / videoRef.duration) * 100 : 0;

	return (
		<div
			className="absolute bottom-0 left-0 w-full bg-black/60 backdrop-blur-sm text-white px-4 py-2 flex items-center space-x-4 z-10"
			onPointerDown={(e) => e.stopPropagation()}
			onPointerUp={(e) => e.stopPropagation()}
			onPointerMove={(e) => e.stopPropagation()}
		>
			<button onClick={togglePlay} className="p-1 hover:text-primary transition cursor-pointer">
				{playing ? <LucidePause /> : <LucidePlay />}
			</button>
			<div className="flex-1 h-2 bg-white/30 rounded cursor-pointer relative" onClick={handleSeek}>
				<div className="h-2 bg-primary rounded pointer-events-none" style={{ width: `${progress}%` }} />
				<div className="absolute top-1/2 -translate-y-1/2" style={{ left: `calc(${progress}% - 6px)` }}>
					<div className="w-3 h-3 bg-primary rounded-full pointer-events-none border-2 border-white shadow" />
				</div>
			</div>
			<span className="text-sm tabular-nums">
				{formatTime(currentTime)} / {videoRef ? formatTime(videoRef.duration) : '00:00'}
			</span>
			<button onClick={handleFullscreen} className="p-1 hover:text-primary transition cursor-pointer">
				<LucideMaximize />
			</button>
		</div>
	);
};

/**
 * Distortion sliders for one lens half.
 */
const LensControls = ({ label, values, onChange }) => {
	const coeffLabels = ['k1', 'k2', 'k3', 'k4'];
	// k1 needs wider range for barrel correction; higher-order terms need less
	const ranges = [
		{ min: -2, max: 2, step: 0.01 },
		{ min: -1, max: 1, step: 0.01 },
		{ min: -0.5, max: 0.5, step: 0.001 },
		{ min: -0.5, max: 0.5, step: 0.001 },
	];
	return (
		<div className="space-y-2">
			<span className="text-sm font-medium">{label}</span>
			<div className="grid grid-cols-2 md:grid-cols-4 gap-3">
				{coeffLabels.map((name, i) => (
					<div key={name} className="space-y-1">
						<div className="flex justify-between text-xs">
							<span>{name}</span>
							<span className="tabular-nums">{values[i].toFixed(3)}</span>
						</div>
						<Slider
							value={[values[i]]}
							onValueChange={([v]) => {
								const next = [...values];
								next[i] = v;
								onChange(next);
							}}
							min={ranges[i].min}
							max={ranges[i].max}
							step={ranges[i].step}
						/>
					</div>
				))}
			</div>
		</div>
	);
};

/**
 * Main panorama viewer page.
 */
export default function PanoramaViewer() {
	// Auto-load video from ?src= query param (e.g. ?src=/dahua_sample.mp4)
	const initialSrc = new URLSearchParams(window.location.search).get('src');
	const [videoSrc, setVideoSrc] = useState(initialSrc);
	const [selectedProfile, setSelectedProfile] = useState(initialSrc ? 'dahua_b180' : 'none');
	const [yawRange, setYawRange] = useState(120);
	const [pitchRange, setPitchRange] = useState(30);
	const containerRef = useRef(null);
	const fileInputRef = useRef(null);

	// Distortion overrides (initialized from the default profile)
	const initialProfile = profiles[initialSrc ? 'dahua_b180' : 'none'];
	const [useCustomD, setUseCustomD] = useState(false);
	const [leftD, setLeftD] = useState([...initialProfile.left.d]);
	const [rightD, setRightD] = useState([...initialProfile.right.d]);
	const [splitPoint, setSplitPoint] = useState(initialProfile.splitPoint);
	const [blendWidth, setBlendWidth] = useState(initialProfile.blendWidth);
	const [fxOverride, setFxOverride] = useState(initialProfile.left.fx);
	const [fyOverride, setFyOverride] = useState(initialProfile.left.fy);
	const [linkHalves, setLinkHalves] = useState(true);

	const profile = profiles[selectedProfile];
	const activeProfile = useCustomD
		? {
				...profile,
				splitPoint,
				blendWidth,
				left: {
					...profile.left,
					fx: fxOverride,
					fy: fyOverride,
					d: leftD,
				},
				right: {
					...profile.right,
					fx: fxOverride,
					fy: fyOverride,
					d: linkHalves ? leftD : rightD,
				},
			}
		: profile;

	// Sync overrides when changing profile
	const handleProfileChange = (key) => {
		setSelectedProfile(key);
		const p = profiles[key];
		if (p) {
			setLeftD([...p.left.d]);
			setRightD([...p.right.d]);
			setSplitPoint(p.splitPoint);
			setBlendWidth(p.blendWidth);
			setFxOverride(p.left.fx);
			setFyOverride(p.left.fy);
		}
	};

	const handleLeftDChange = (d) => {
		setLeftD(d);
		if (linkHalves) setRightD(d);
	};

	const handleFileSelect = useCallback(
		(e) => {
			const file = e.target.files?.[0];
			if (file) {
				if (videoSrc && videoSrc.startsWith('blob:')) {
					URL.revokeObjectURL(videoSrc);
				}
				setVideoSrc(URL.createObjectURL(file));
			}
		},
		[videoSrc]
	);

	useEffect(() => {
		return () => {
			if (videoSrc && videoSrc.startsWith('blob:')) {
				URL.revokeObjectURL(videoSrc);
			}
		};
	}, [videoSrc]);

	const setSelectedMatch = useViewerStore((s) => s.setSelectedMatch);
	useEffect(() => {
		if (videoSrc) {
			setSelectedMatch({ id: 'panorama', src: videoSrc });
		}
	}, [videoSrc, setSelectedMatch]);

	if (!videoSrc) {
		return (
			<div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 p-8">
				<h1 className="text-2xl font-bold">Panorama Viewer</h1>
				<p className="text-muted-foreground text-center max-w-lg">
					Load a raw panoramic video from your Dahua or Reolink camera to view it with real-time dewarping and
					pan/zoom controls.
				</p>
				<Button onClick={() => fileInputRef.current?.click()} size="lg">
					<LucideUpload className="mr-2 h-5 w-5" />
					Open Video File
				</Button>
				<input ref={fileInputRef} type="file" accept="video/*" onChange={handleFileSelect} className="hidden" />
			</div>
		);
	}

	return (
		<div className="w-full flex flex-col items-center gap-4 px-4 py-4">
			<h1 className="text-xl font-bold">Panorama Viewer</h1>

			{/* Video viewer */}
			<ErrorBoundary fallback={<div className="text-red-500">Failed to load viewer</div>}>
				<CameraProvider>
					<CameraControlsWrapper yawRange={yawRange} pitchRange={pitchRange}>
						<div
							ref={containerRef}
							className="relative w-full max-w-6xl bg-black overflow-hidden rounded-lg"
						>
							<div className="w-full h-full aspect-video">
								<Canvas
									frameloop="always"
									camera={{
										position: [0, 0, 0],
										fov: 35,
										near: 0.01,
										far: 5,
									}}
								>
									<PanoramaControls />
									<PanoramaScene videoSrc={videoSrc} profile={activeProfile} />
								</Canvas>
							</div>
							<PlayerControls containerRef={containerRef} />
						</div>
					</CameraControlsWrapper>
				</CameraProvider>
			</ErrorBoundary>

			{/* Settings panel */}
			<div className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-6 p-4 bg-card rounded-lg border">
				{/* Camera profile */}
				<div className="space-y-3">
					<Label className="text-sm font-semibold">Camera Profile</Label>
					<select
						value={selectedProfile}
						onChange={(e) => handleProfileChange(e.target.value)}
						className="w-full px-3 py-2 border rounded-md bg-background"
					>
						{Object.entries(profiles).map(([key, p]) => (
							<option key={key} value={key}>
								{p.label}
							</option>
						))}
					</select>
					<Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
						<LucideUpload className="mr-2 h-4 w-4" />
						Load Different Video
					</Button>
					<input
						ref={fileInputRef}
						type="file"
						accept="video/*"
						onChange={handleFileSelect}
						className="hidden"
					/>
				</div>

				{/* Pan/zoom ranges */}
				<div className="space-y-3">
					<Label className="text-sm font-semibold">Pan/Zoom Range</Label>
					<div className="space-y-2">
						<div className="flex justify-between text-sm">
							<span>Horizontal (yaw)</span>
							<span>{yawRange} deg</span>
						</div>
						<Slider
							value={[yawRange]}
							onValueChange={([v]) => setYawRange(v)}
							min={60}
							max={180}
							step={5}
						/>
					</div>
					<div className="space-y-2">
						<div className="flex justify-between text-sm">
							<span>Vertical (pitch)</span>
							<span>{pitchRange} deg</span>
						</div>
						<Slider
							value={[pitchRange]}
							onValueChange={([v]) => setPitchRange(v)}
							min={10}
							max={90}
							step={5}
						/>
					</div>
				</div>

				{/* Distortion tuning */}
				<div className="space-y-3 md:col-span-2">
					<div className="flex items-center gap-4">
						<Label className="text-sm font-semibold">Distortion Coefficients</Label>
						<label className="flex items-center gap-2 text-sm">
							<input
								type="checkbox"
								checked={useCustomD}
								onChange={(e) => setUseCustomD(e.target.checked)}
							/>
							Override
						</label>
						{useCustomD && (
							<label className="flex items-center gap-2 text-sm">
								<input
									type="checkbox"
									checked={linkHalves}
									onChange={(e) => {
										setLinkHalves(e.target.checked);
										if (e.target.checked) setRightD([...leftD]);
									}}
								/>
								Link L/R
							</label>
						)}
					</div>
					{useCustomD && (
						<div className="space-y-4">
							{/* Focal length and seam controls */}
							<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
								<div className="space-y-1">
									<div className="flex justify-between text-xs">
										<span>fx (scale)</span>
										<span className="tabular-nums">{fxOverride}</span>
									</div>
									<Slider
										value={[fxOverride]}
										onValueChange={([v]) => setFxOverride(v)}
										min={200}
										max={4000}
										step={50}
									/>
								</div>
								<div className="space-y-1">
									<div className="flex justify-between text-xs">
										<span>fy (scale)</span>
										<span className="tabular-nums">{fyOverride}</span>
									</div>
									<Slider
										value={[fyOverride]}
										onValueChange={([v]) => setFyOverride(v)}
										min={200}
										max={4000}
										step={50}
									/>
								</div>
								<div className="space-y-1">
									<div className="flex justify-between text-xs">
										<span>Split point</span>
										<span className="tabular-nums">{splitPoint.toFixed(2)}</span>
									</div>
									<Slider
										value={[splitPoint]}
										onValueChange={([v]) => setSplitPoint(v)}
										min={0.3}
										max={0.7}
										step={0.01}
									/>
								</div>
								<div className="space-y-1">
									<div className="flex justify-between text-xs">
										<span>Blend width</span>
										<span className="tabular-nums">{blendWidth.toFixed(3)}</span>
									</div>
									<Slider
										value={[blendWidth]}
										onValueChange={([v]) => setBlendWidth(v)}
										min={0}
										max={0.1}
										step={0.002}
									/>
								</div>
							</div>

							{/* Left lens */}
							<LensControls
								label={linkHalves ? 'Both Lenses' : 'Left Lens'}
								values={leftD}
								onChange={handleLeftDChange}
							/>

							{/* Right lens (only when not linked) */}
							{!linkHalves && <LensControls label="Right Lens" values={rightD} onChange={setRightD} />}
						</div>
					)}
				</div>
			</div>
		</div>
	);
}
