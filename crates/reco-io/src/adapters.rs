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
/// paired by arrival order (assumes same-length files from the same
/// recording session).
#[cfg(feature = "ffmpeg")]
pub struct FfmpegFileSource {
    rx: std::sync::mpsc::Receiver<FramePair>,
    info: SourceInfo,
    decode_backend: ffmpeg::decoder::DecodeBackend,
}

#[cfg(feature = "ffmpeg")]
impl FfmpegFileSource {
    /// Open a stereo file source from two video file paths.
    pub fn open(
        left_path: &std::path::Path,
        right_path: &std::path::Path,
    ) -> Result<Self, SourceError> {
        let probe = ffmpeg::decoder::VideoDecoder::open(left_path)
            .map_err(|e| SourceError::Init(format!("left: {e}")))?;
        let info = SourceInfo {
            width: probe.width(),
            height: probe.height(),
            fps: probe.fps(),
        };
        let decode_backend = probe.backend();
        drop(probe);

        let left = left_path.to_path_buf();
        let right = right_path.to_path_buf();
        let rx = Self::spawn_decode_pipeline(left, right);

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

    /// Returns the FFmpeg frame rate rational (num/den) for encoder setup.
    ///
    /// Re-probes the left file. Call once during setup, not per-frame.
    pub fn frame_rate(left_path: &std::path::Path) -> Result<ffmpeg_next::Rational, SourceError> {
        let dec = ffmpeg::decoder::VideoDecoder::open(left_path)
            .map_err(|e| SourceError::Init(format!("{e}")))?;
        Ok(dec.frame_rate())
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
    ) -> std::sync::mpsc::Receiver<FramePair> {
        let left_rx = Self::spawn_single_decoder(left_path, "left");
        let right_rx = Self::spawn_single_decoder(right_path, "right");

        let (tx, rx) = std::sync::mpsc::sync_channel::<FramePair>(4);

        std::thread::Builder::new()
            .name("decode_pair".into())
            .spawn(move || {
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
        fps: ffmpeg_next::Rational,
        config: &ffmpeg::encoder::EncoderConfig,
    ) -> Result<Self, EncodeError> {
        let inner = ffmpeg::encoder::VideoEncoder::new(path, width, height, fps, config)
            .map_err(|e| EncodeError::Init(e.to_string()))?;
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
            PixelFormat::Nv12 => self
                .inner
                .write_nv12_frame(frame.data)
                .map_err(|e| EncodeError::Frame(e.to_string())),
            PixelFormat::Rgba8 => self
                .inner
                .write_frame(frame.data)
                .map_err(|e| EncodeError::Frame(e.to_string())),
        }
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        self.inner
            .finish()
            .map_err(|e| EncodeError::Finalize(e.to_string()))
    }
}
