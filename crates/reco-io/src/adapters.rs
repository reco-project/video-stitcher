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
}

#[cfg(feature = "ffmpeg")]
impl FfmpegFileSource {
    /// Open a stereo file source from two video file paths (no sync offset).
    pub fn open(
        left_path: &std::path::Path,
        right_path: &std::path::Path,
    ) -> Result<Self, SourceError> {
        Self::open_with_offset(left_path, right_path, 0)
    }

    /// Open a stereo file source with a temporal sync offset.
    ///
    /// `sync_offset` specifies how many frames to skip for alignment:
    /// - Positive: skip N frames from the **right** video (right started first)
    /// - Negative: skip N frames from the **left** video (left started first)
    /// - Zero: pair by arrival order (no offset)
    pub fn open_with_offset(
        left_path: &std::path::Path,
        right_path: &std::path::Path,
        sync_offset: i64,
    ) -> Result<Self, SourceError> {
        let probe =
            ffmpeg::decoder::VideoDecoder::open(left_path).map_err(|e| SourceError::Init {
                path: left_path.display().to_string(),
                reason: format!("{e}"),
            })?;
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
        };
        let decode_backend = probe.backend();
        drop(probe);

        let left = left_path.to_path_buf();
        let right = right_path.to_path_buf();
        let rx = Self::spawn_decode_pipeline(left, right, sync_offset);

        Ok(Self {
            rx,
            info,
            decode_backend,
        })
    }

    /// The decode backend selected during probe (CUDA, VAAPI, or software).
    pub fn decode_backend(&self) -> ffmpeg::decoder::DecodeBackend {
        self.decode_backend
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

    fn spawn_single_decoder(
        path: std::path::PathBuf,
        label: &'static str,
    ) -> std::sync::mpsc::Receiver<YuvData> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<YuvData>(4);

        std::thread::Builder::new()
            .name(format!("decode_{label}"))
            .spawn(move || {
                let mut dec = match ffmpeg::decoder::VideoDecoder::open(&path) {
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

    fn spawn_decode_pipeline(
        left_path: std::path::PathBuf,
        right_path: std::path::PathBuf,
        sync_offset: i64,
    ) -> std::sync::mpsc::Receiver<FramePair> {
        let left_rx = Self::spawn_single_decoder(left_path, "left");
        let right_rx = Self::spawn_single_decoder(right_path, "right");

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
        SourceInfo {
            width: self.info.width,
            height: self.info.height,
            fps: self.info.fps,
        }
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.recv() {
            Ok(pair) => Ok(Some(StereoFrame::Yuv420p(pair))),
            Err(_) => Ok(None),
        }
    }

    fn try_next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        match self.rx.try_recv() {
            Ok(pair) => Ok(Some(StereoFrame::Yuv420p(pair))),
            Err(std::sync::mpsc::TryRecvError::Empty) => Ok(None),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Ok(None),
        }
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
#[cfg(feature = "ffmpeg")]
pub fn create_encoder(
    path: &std::path::Path,
    width: u32,
    height: u32,
    fps: (i32, i32),
    codec: &str,
    quality: &str,
    encoder_name: Option<String>,
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
