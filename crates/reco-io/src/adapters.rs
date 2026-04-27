//! Trait adapters: wrap backend-specific types into `reco_core` traits.
//!
//! These are thin wrappers that bridge the gap between backend APIs
//! (FFmpeg, GStreamer) and the `FrameSource`/`Encoder` traits defined
//! in `reco-core`. Backend code stays clean and trait-free; all trait
//! plumbing lives here.

use reco_core::encoder::{EncodeError, Encoder, OutputFrame, PixelFormat};
use reco_core::source::{FramePair, SourceError, SourceInfo, StereoFrame, YuvData};

#[cfg(feature = "ffmpeg")]
use crate::ffmpeg;

// -- FFmpeg File Source --

/// Stereo file source backed by two FFmpeg decoders.
///
/// Opens two video files (left + right camera) and delivers synchronized
/// YUV420P frame pairs. Each decoder runs in its own thread; frames are
/// paired after applying a temporal sync offset.
///
/// ## Sync Offset
///
/// When cameras don't start recording at the same instant, a frame offset
/// aligns them temporally. Use [`Self::open_with_offset`]:
/// - Positive offset: skip N frames from the **right** video (right started first)
/// - Negative offset: skip N frames from the **left** video (left started first)
#[cfg(feature = "ffmpeg")]
pub struct FfmpegFileSource {
    rx: std::sync::mpsc::Receiver<FramePair>,
    info: SourceInfo,
    decode_backend: ffmpeg::decoder::DecodeBackend,
    /// GPU pixel format (NV12 8-bit or P010 10-bit).
    pixel_format: reco_core::renderer::GpuPixelFormat,
    /// Rotation from stream metadata (0, 90, 180, 270).
    left_rotation: i32,
    right_rotation: i32,
    left_input: crate::stitch_job::InputPath,
    right_input: crate::stitch_job::InputPath,
    sync_offset: i64,
    /// Total frame count (estimated from duration * fps).
    total_frame_count: Option<u64>,
    /// Current frame position (incremented on each next_frame).
    current_frame: u64,
    /// True once the decode pipeline has signaled end-of-stream.
    ///
    /// Set when the internal receiver reports `Disconnected` (both decode
    /// threads have finished). A blocking `recv()` returning `Err` or a
    /// non-blocking `try_recv()` returning `Disconnected` are the two
    /// places this becomes `true`. Reset by `seek()`, which respawns
    /// the decode pipeline.
    exhausted: bool,
}

#[cfg(feature = "ffmpeg")]
impl FfmpegFileSource {
    pub fn open(
        left_path: &std::path::Path,
        right_path: &std::path::Path,
    ) -> Result<Self, SourceError> {
        Self::open_with_offset(left_path, right_path, 0)
    }

    pub fn open_with_offset(
        left_path: &std::path::Path,
        right_path: &std::path::Path,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        Self::open_from_inputs(
            &crate::stitch_job::InputPath::Single(left_path.to_path_buf()),
            &crate::stitch_job::InputPath::Single(right_path.to_path_buf()),
            sync_offset,
        )
    }

    /// Open from `InputPath` to support chained multi-segment files.
    ///
    /// Accepts `InputPath` to support chained multi-segment files
    /// (GoPro/DJI auto-split). Probing uses the first segment.
    pub fn open_from_inputs(
        left: &crate::stitch_job::InputPath,
        right: &crate::stitch_job::InputPath,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let left_probe_path = left.first_path();
        let right_probe_path = right.first_path();

        reco_core::source::validate_input_path(left_probe_path)?;
        reco_core::source::validate_input_path(right_probe_path)?;

        let probe = ffmpeg::decoder::VideoDecoder::open(left_probe_path).map_err(|e| {
            SourceError::Init {
                path: left_probe_path.display().to_string(),
                reason: format!("{e}"),
            }
        })?;
        let fps_r = probe.frame_rate();
        let fps = probe.fps();
        let total_frame_count = probe.duration_secs().map(|dur| (dur * fps) as u64);
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps,
            fps_rational: Some((fps_r.0, fps_r.1)),
            total_frames: total_frame_count,
        };
        let decode_backend = probe.backend();
        let pixel_format = probe.pixel_format();
        let left_rotation = probe.rotation();
        drop(probe);

        let right_rotation = ffmpeg::decoder::VideoDecoder::open(right_probe_path)
            .map(|d| d.rotation())
            .unwrap_or_else(|e| {
                log::warn!("Failed to probe right video for rotation ({e}), assuming 0 degrees");
                0
            });

        let left_owned = left.clone();
        let right_owned = right.clone();
        let rx = Self::spawn_decode_pipeline_from_inputs(
            left_owned.clone(),
            right_owned.clone(),
            sync_offset,
            None,
        );

        Ok(Self {
            rx,
            info,
            decode_backend,
            pixel_format,
            left_rotation,
            right_rotation,
            left_input: left_owned,
            right_input: right_owned,
            sync_offset,
            total_frame_count,
            current_frame: 0,
            exhausted: false,
        })
    }

    /// The decode backend selected during probe (CUDA, VAAPI, or software).
    pub fn decode_backend(&self) -> ffmpeg::decoder::DecodeBackend {
        self.decode_backend
    }

    /// GPU pixel format for zero-copy shared textures.
    ///
    /// Returns `GpuPixelFormat::P010` for 10-bit sources or
    /// `GpuPixelFormat::Nv12` for 8-bit.
    pub fn pixel_format(&self) -> reco_core::renderer::GpuPixelFormat {
        self.pixel_format
    }

    /// Left stream rotation from metadata (0, 90, 180, 270 degrees).
    pub fn left_rotation(&self) -> i32 {
        self.left_rotation
    }

    /// Right stream rotation from metadata.
    pub fn right_rotation(&self) -> i32 {
        self.right_rotation
    }

    /// Whether this source's decode backend supports zero-copy GPU transfer.
    ///
    /// Returns `true` if the decoder uses a hardware path that can write
    /// directly to GPU-shared memory (CUDA on Linux, VideoToolbox on macOS).
    pub fn supports_zero_copy(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            self.decode_backend == ffmpeg::decoder::DecodeBackend::Cuda
        }
        #[cfg(target_os = "macos")]
        {
            self.decode_backend == ffmpeg::decoder::DecodeBackend::VideoToolbox
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            false
        }
    }

    /// Returns the frame rate as `(numerator, denominator)` for encoder setup.
    ///
    /// Re-probes the left file. Call once during setup, not per-frame.
    pub fn frame_rate(left_path: &std::path::Path) -> Result<(i32, i32), SourceError> {
        let dec =
            ffmpeg::decoder::VideoDecoder::open(left_path).map_err(|e| SourceError::Init {
                path: left_path.display().to_string(),
                reason: format!("{e}"),
            })?;
        let r = dec.frame_rate();
        Ok((r.0, r.1))
    }

    fn spawn_single_decoder_at(
        input: crate::stitch_job::InputPath,
        label: &'static str,
        seek_secs: Option<f64>,
    ) -> std::sync::mpsc::Receiver<YuvData> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<YuvData>(4);

        std::thread::Builder::new()
            .name(format!("decode_{label}"))
            .spawn(move || {
                let mut dec = match ffmpeg::decoder::VideoDecoder::open_input(&input) {
                    Ok(d) => {
                        log::info!(
                            "{label} decoder: {} ({}x{})",
                            d.backend(),
                            d.width(),
                            d.height()
                        );
                        d
                    }
                    Err(e) => {
                        log::error!("Failed to open {label} video: {e}");
                        return;
                    }
                };
                if let Some(secs) = seek_secs {
                    if let Err(e) = dec.seek_to_secs(secs) {
                        log::error!("{label} seek to {secs:.1}s failed: {e}");
                        return;
                    }
                    // FFmpeg seeks to the nearest keyframe BEFORE the target.
                    // Decode and discard frames until we reach the target PTS.
                    let target_us = (secs * 1_000_000.0) as i64;
                    let mut skipped = 0u32;
                    loop {
                        match dec.next_frame() {
                            Ok(Some(f)) if f.timestamp_us < target_us => {
                                skipped += 1;
                            }
                            Ok(Some(f)) => {
                                log::debug!(
                                    "{label} seek: skipped {skipped} frames to reach target"
                                );
                                let buf = YuvData {
                                    y: f.y,
                                    u: f.u,
                                    v: f.v,
                                };
                                if tx.send(buf).is_err() {
                                    return;
                                }
                                break;
                            }
                            _ => return,
                        }
                    }
                }
                loop {
                    match dec.next_frame() {
                        Ok(Some(f)) => {
                            let buf = YuvData {
                                y: f.y,
                                u: f.u,
                                v: f.v,
                            };
                            if tx.send(buf).is_err() {
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            log::error!("{label} decode error: {e}");
                            break;
                        }
                    }
                }
            })
            .expect("spawn decode thread");

        rx
    }

    fn spawn_decode_pipeline_from_inputs(
        left: crate::stitch_job::InputPath,
        right: crate::stitch_job::InputPath,
        sync_offset: i64,
        seek_secs: Option<f64>,
    ) -> std::sync::mpsc::Receiver<FramePair> {
        let left_rx = Self::spawn_single_decoder_at(left, "left", seek_secs);
        let right_rx = Self::spawn_single_decoder_at(right, "right", seek_secs);

        let (tx, rx) = std::sync::mpsc::sync_channel::<FramePair>(4);

        std::thread::Builder::new()
            .name("decode_pair".into())
            .spawn(move || {
                // Apply sync offset: skip frames from the camera that started first.
                if sync_offset > 0 {
                    // Right started first — skip N right frames.
                    for _ in 0..sync_offset {
                        if right_rx.recv().is_err() {
                            return;
                        }
                    }
                    log::info!("Sync offset: skipped {sync_offset} right frames");
                } else if sync_offset < 0 {
                    // Left started first — skip N left frames.
                    let skip = sync_offset.unsigned_abs();
                    for _ in 0..skip {
                        if left_rx.recv().is_err() {
                            return;
                        }
                    }
                    log::info!("Sync offset: skipped {skip} left frames");
                }

                while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                    if tx.send(FramePair { left, right }).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn pairing thread");

        rx
    }
}

#[cfg(feature = "ffmpeg")]
impl reco_core::source::FrameSource for FfmpegFileSource {
    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn left_rotation(&self) -> i32 {
        self.left_rotation
    }

    fn right_rotation(&self) -> i32 {
        self.right_rotation
    }

    fn gpu_pixel_format(&self) -> reco_core::renderer::GpuPixelFormat {
        self.pixel_format
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => {
                self.current_frame += 1;
                Ok(Some(StereoFrame::Yuv420p(pair)))
            }
            Err(_) => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }

    fn try_next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.try_recv() {
            Ok(pair) => {
                self.current_frame += 1;
                Ok(Some(StereoFrame::Yuv420p(pair)))
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.exhausted = true;
                Ok(None)
            }
        }
    }

    fn seek(&mut self, frame: u64) -> Result<(), SourceError> {
        // Two strategies depending on seek direction and distance:
        //
        // 1. **Forward ≤ 10s**: drain frames from the existing decode
        //    pipeline. The decode threads are already running and
        //    buffering, so this is near-instant with no NVDEC re-init.
        //
        // 2. **Backward or > 10s forward**: respawn the entire decode
        //    pipeline at the target position. Each decoder seeks to the
        //    nearest keyframe before the target, then decodes and
        //    discards frames until reaching the exact target PTS.
        //
        // Callers handling rapid input (scrub bars, key repeat) should
        // coalesce seek requests on their side and call this once with
        // the final target. This method is blocking for strategy 1 and
        // semi-blocking for strategy 2 (spawns threads, returns before
        // first frame is decoded).
        if frame > self.current_frame {
            let skip = frame - self.current_frame;
            let max_forward = (self.info.fps * 10.0) as u64;
            if skip <= max_forward {
                log::debug!("Forward seek: skipping {skip} frames via decode");
                for _ in 0..skip {
                    match self.rx.recv() {
                        Ok(_) => self.current_frame += 1,
                        Err(_) => break,
                    }
                }
                return Ok(());
            }
        }

        let secs = frame as f64 / self.info.fps;
        log::info!("Seeking to frame {frame} ({secs:.1}s)");
        // Drop old receiver - disconnects current decode threads which
        // exit on the next send() failure. Spawns fresh decoders that
        // seek to the target keyframe and skip to the exact PTS.
        self.rx = Self::spawn_decode_pipeline_from_inputs(
            self.left_input.clone(),
            self.right_input.clone(),
            self.sync_offset,
            Some(secs),
        );
        self.current_frame = frame;
        self.exhausted = false;
        Ok(())
    }

    fn total_frames(&self) -> Option<u64> {
        self.total_frame_count
    }

    fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

// -- Zero-copy detection helper --

/// Detect whether the zero-copy GPU pipeline should be used.
///
/// Checks three conditions:
/// 1. The `RECO_NO_HWACCEL` environment variable is **not** set
/// 2. The source's decode backend supports zero-copy (CUDA on Linux, VideoToolbox on macOS)
/// 3. The GPU context supports zero-copy interop (Vulkan+CUDA on Linux, Metal on macOS)
///
/// Returns `true` only if all three conditions are met.
#[cfg(feature = "ffmpeg")]
pub fn detect_zero_copy(source: &FfmpegFileSource, gpu: &reco_core::gpu::GpuContext) -> bool {
    std::env::var("RECO_NO_HWACCEL").is_err()
        && source.supports_zero_copy()
        && gpu.supports_zero_copy()
}

// -- Encoder creation helper --

/// Create an FFmpeg file encoder from high-level parameters.
///
/// Wraps codec parsing, quality mapping, and encoder creation into a single
/// call. Returns the encoder and the name of the selected encoder backend
/// (e.g. `"h264_nvenc"`, `"libx264"`).
///
/// This is the preferred way for consumers (CLI, GUI, cloud) to create an
/// encoder without duplicating codec/quality parsing logic.
///
/// # Arguments
///
/// * `path` - Output file path.
/// * `width`, `height` - Output frame dimensions.
/// * `fps` - Frame rate as `(numerator, denominator)`.
/// * `codec` - Codec name: `"h264"`, `"hevc"`, or `"av1"`.
/// * `quality` - Quality preset: `"fast"`, `"balanced"`, or `"high"`.
/// * `encoder_name` - Force a specific encoder by name, or `None` for auto-detection.
/// * `crf` - Override the CRF/quality value, or `None` to use the quality tier default.
/// * `preset` - Override the encoder preset string, or `None` to use the quality tier default.
#[cfg(feature = "ffmpeg")]
#[allow(clippy::too_many_arguments)]
pub fn create_encoder(
    path: &std::path::Path,
    width: u32,
    height: u32,
    fps: (i32, i32),
    codec: &str,
    quality: &str,
    encoder_name: Option<String>,
    crf: Option<u8>,
    preset: Option<String>,
) -> Result<(FfmpegFileEncoder, String), reco_core::encoder::EncodeError> {
    let quality_enum = match quality {
        "fast" => ffmpeg::encoder::Quality::Fast,
        "high" => ffmpeg::encoder::Quality::High,
        _ => ffmpeg::encoder::Quality::Balanced,
    };
    let video_codec = ffmpeg::encoder::VideoCodec::from_str_loose(codec).unwrap_or_else(|| {
        log::warn!("Unknown codec '{codec}', defaulting to H.264");
        ffmpeg::encoder::VideoCodec::H264
    });
    let enc_config = ffmpeg::encoder::EncoderConfig {
        encoder_name,
        codec: video_codec,
        quality: quality_enum,
        crf,
        preset,
        audio_source: None,
        container: ffmpeg::encoder::Container::default(),
        gop_size: None,
    };
    let encoder = FfmpegFileEncoder::new(path, width, height, fps, &enc_config)?;
    let name = encoder.encoder_name().to_string();
    Ok((encoder, name))
}

// -- FFmpeg File Encoder --

/// File encoder backed by FFmpeg.
///
/// Thin wrapper around `ffmpeg::encoder::VideoEncoder` that implements
/// the `reco_core::encoder::Encoder` trait.
#[cfg(feature = "ffmpeg")]
pub struct FfmpegFileEncoder {
    inner: ffmpeg::encoder::VideoEncoder,
}

#[cfg(feature = "ffmpeg")]
impl FfmpegFileEncoder {
    /// Create a new file encoder.
    pub fn new(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps: (i32, i32),
        config: &ffmpeg::encoder::EncoderConfig,
    ) -> Result<Self, EncodeError> {
        let fps_rational = ffmpeg_next::Rational(fps.0, fps.1);
        let inner = ffmpeg::encoder::VideoEncoder::new(path, width, height, fps_rational, config)
            .map_err(|e| EncodeError::Init {
            reason: e.to_string(),
        })?;
        Ok(Self { inner })
    }

    /// The name of the active encoder (e.g. "libx264", "h264_nvenc").
    pub fn encoder_name(&self) -> &str {
        self.inner.encoder_name()
    }
}

#[cfg(feature = "ffmpeg")]
impl Encoder for FfmpegFileEncoder {
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError> {
        match frame.format {
            PixelFormat::Nv12 => {
                self.inner
                    .write_nv12_frame(frame.data)
                    .map_err(|e| EncodeError::Frame {
                        frame_index: None,
                        reason: e.to_string(),
                    })
            }
            PixelFormat::Rgba8 => {
                self.inner
                    .write_frame(frame.data)
                    .map_err(|e| EncodeError::Frame {
                        frame_index: None,
                        reason: e.to_string(),
                    })
            }
        }
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        self.inner.finish().map_err(|e| EncodeError::Finalize {
            reason: e.to_string(),
        })
    }
}
