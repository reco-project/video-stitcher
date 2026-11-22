//! Reco CLI — panoramic video stitching from the command line.
//!
//! ```text
//! reco stitch left.mp4 right.mp4 --calibration match.json -o output.mp4
//! ```

mod calibrate;
#[cfg(feature = "gstreamer")]
mod camera;
mod helpers;
#[cfg(feature = "libcamera")]
mod libcamera_cmd;
mod preview;
#[cfg(feature = "profiling")]
mod profiling;
#[cfg(feature = "snapshot")]
mod snapshot;
mod stitch;

use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Install the standard tracing subscriber for the non-profiling path.
///
/// Routes legacy `log::*` macro calls from reco-core / reco-io /
/// reco-calibrate into the tracing pipeline so a single structured
/// source of truth carries every event. Reads `RUST_LOG` for level
/// filtering; defaults to `info` if unset.
#[cfg(not(feature = "profiling"))]
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt, prelude::*};

    // Bridge legacy `log::*` calls into tracing. Ignores "already set"
    // errors so tests that construct multiple CLIs don't panic.
    let _ = tracing_log::LogTracer::init();

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort::logging=warn"));
    let fmt_layer = fmt::layer().with_target(true).with_level(true);

    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init();
}

#[cfg(target_os = "linux")]
fn quiet_libva_info_by_default() {
    if std::env::var_os("LIBVA_MESSAGING_LEVEL").is_none() {
        // SAFETY: This runs at CLI process startup, before we install the
        // Ctrl-C handler or start decode/encode worker threads. Callers can
        // still opt back into libva info logs by setting the env var.
        unsafe {
            std::env::set_var("LIBVA_MESSAGING_LEVEL", "1");
        }
    }
}

/// Install a panic hook that captures the panic context (location,
/// payload) as a structured `tracing::error!` event before the default
/// hook runs. When a user reports a bug post-deployment with a log
/// file, the panic context is immediately searchable alongside regular
/// log lines.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".into());
        let payload = if let Some(s) = info.payload().downcast_ref::<&str>() {
            (*s).to_string()
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "<non-string panic payload>".into()
        };
        tracing::error!(
            target: "panic",
            location = %location,
            payload = %payload,
            "panic caught by tracing panic hook"
        );
        default_hook(info);
    }));
}

#[derive(Parser)]
#[command(
    name = "reco",
    version,
    about = "GPU-accelerated panoramic video stitching",
    long_about = "Reco stitches two camera feeds into a seamless panoramic sports view.\n\
                  Designed for sports filming with open-source hardware flexibility."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
// CLI subcommand variants are inherently size-imbalanced (Stitch carries
// many optional args); boxing individual fields would only obscure the
// flag mapping. The enum is constructed once at startup, so the size gap
// is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Commands {
    /// Stitch camera video files or RTSP feeds into a panoramic output.
    Stitch {
        /// Input videos, one per camera in projection order (two for
        /// L-shape or panorama calibrations, one pre-stitched panorama for the
        /// cylinder). Chain multi-segment recordings per input with
        /// `;` (e.g. `a1.mp4;a2.mp4 b1.mp4;b2.mp4`). RTSP inputs require --rtsp-fps and --rtsp-offsets-us.
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<String>,

        #[command(flatten)]
        live_input: stitch::input::LiveInputArgs,

        /// Copy audio from this zero-based input index, or 'none' to disable source audio. No mixing or transcoding.
        #[arg(long, default_value = "0", value_parser = stitch::input::parse_audio_source)]
        audio_source: reco_io::output::AudioMode,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long, default_value = "output.mp4")]
        output: String,

        /// Also publish H.264 + AAC/MP3 to RTMP/RTMPS, paced at media speed. Keeps the recording if streaming fails.
        #[arg(long)]
        stream_url: Option<String>,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Start processing at this time offset (seconds).
        #[arg(long)]
        start_time: Option<f64>,

        /// Stop processing at this time offset (seconds).
        #[arg(long)]
        end_time: Option<f64>,

        /// Hard cap on the number of output frames.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Force a specific encoder (e.g., h264_nvenc, hevc_nvenc, libx264). Auto-detects by default.
        #[arg(long)]
        encoder: Option<String>,

        /// VAAPI render node or NVENC GPU index. Requires --encoder.
        #[arg(long, requires = "encoder")]
        encoder_device: Option<String>,

        /// Output codec: h264, hevc, av1. Defaults to the named encoder's codec, or h264.
        #[arg(long)]
        codec: Option<String>,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "balanced")]
        quality: String,

        /// Seam blend width (0.0–1.0). 0 = hard edge. Overrides the
        /// calibration's saved value; when omitted, the calibration decides.
        #[arg(long, value_parser = parse_blend)]
        blend: Option<f32>,

        /// Export a fixed cylindrical panorama from an L-shape calibration (horizontal degrees).
        #[arg(long, requires = "panorama_vertical_fov")]
        panorama_arc: Option<f64>,

        /// Panorama vertical extent in degrees; requires --panorama-arc.
        #[arg(long, requires = "panorama_arc")]
        panorama_vertical_fov: Option<f64>,

        /// Frame offset for temporal sync between cameras.
        /// Positive: skip N right frames (right started first).
        /// Negative: skip N left frames (left started first).
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Path to a YOLO ONNX model for ball detection and auto-panning.
        /// When provided, enables automatic camera direction that follows the ball.
        #[arg(long)]
        model: Option<String>,

        /// Run detection every N frames (default: 1 = every frame).
        /// Higher values reduce detection overhead. The panner uses
        /// the last known detections on skipped frames.
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Lookahead buffer in seconds. Decodes N frames ahead so the
        /// panner can lead the play and the loop can centered-smooth the
        /// pose lag-free. The validated dead-zone/reactivity defaults
        /// assume this is on, so it defaults to 1.5. Only engages with AI
        /// tracking (needs --model, non-sweep); ignored for a plain
        /// stitch. Retains source images on their selected storage path. 0 = off.
        #[arg(long, default_value_t = 1.5)]
        lookahead: f64,

        /// Override retained lookahead image storage in bytes (not total process memory).
        #[arg(long)]
        lookahead_memory_bytes: Option<usize>,

        /// Tracking mode: "field" (ball + players, default), "ball"
        /// (ball only), "sweep" (no AI, debug pan). field is robust with
        /// COCO models; ball-only follows the weak COCO ball alone.
        #[arg(long, default_value = "field", value_parser = ["field", "ball", "sweep"])]
        tracking: String,

        /// Encoder quality on a 0-100 scale (higher = better). Overrides the
        /// quality preset with a precise value. Converted per encoder:
        /// CRF-style encoders map to `crf = 40 - (quality/100)*28`,
        /// VideoToolbox passes through as `global_quality`. Examples:
        /// 80 = high quality, 65 = balanced, 50 = smaller files.
        #[arg(long = "quality-value")]
        quality_value: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264; p1-p7 for NVENC).
        #[arg(long)]
        preset: Option<String>,

        /// Output container format. One of: `mp4` (default,
        /// finalized at close), `fmp4` (fragmented MP4, readable
        /// mid-write), `mkv` (Matroska, crash-safe + streamable).
        /// Pick `mkv` or `fmp4` if you plan to tee the output to
        /// RTMP via a second ffmpeg with `-c copy` while the
        /// stitch is still running.
        #[arg(long)]
        container: Option<String>,

        /// Record pre-stitch source frames to this path as a
        /// stacked-video file for later replay or cloud upload.
        /// Requires building with `--features replay`.
        #[arg(long)]
        replay: Option<String>,

        /// Downscale replay tiles to `WIDTHxHEIGHT` (e.g.
        /// `1280x720`, `854x480`). Reduces replay file size and
        /// CPU encode cost (libx264 at 1080p stacked crawls on
        /// ARM / Jetson). GPU pack path only: CPU-resident
        /// sources log a warn and record at source dims. Width
        /// must be divisible by 4; height must be even. Requires
        /// `--replay`.
        #[arg(long, value_parser = parse_wxh)]
        replay_scale: Option<(u32, u32)>,

        /// Force CPU video decode instead of GPU zero-copy (NVDEC).
        /// Needed when AI tracking uses ORT CPU detection without
        /// TensorRT. Slower but lets the detector see the frames.
        #[arg(long, default_value_t = false)]
        no_zero_copy: bool,

        /// Stitch entirely on the CPU: software render + CPU decode,
        /// no GPU touched. Much slower; for GPU-less machines and for
        /// verifying output without GPU variance.
        #[arg(long, default_value_t = false)]
        cpu: bool,

        /// Record pipeline events (detections, filter decisions, pan
        /// decisions) to a JSONL file for offline analysis.
        #[arg(long)]
        events: Option<String>,

        /// Precomputed trajectory CSV (frame,yaw,pitch,fov). Overrides
        /// AI tracking with poses read from the file.
        #[arg(long)]
        trajectory: Option<String>,

        /// FieldPanner tuning as a JSON file (field mode). Only the keys
        /// present override the base, e.g. {"framing":"frame_all",
        /// "dead_zone_rad":0.087}. Overlays on --panner-preset if both set.
        #[arg(long = "panner-config")]
        panner_config: Option<String>,

        /// Named panner preset: broadcast (default), action, frame_all.
        #[arg(long = "panner-preset")]
        panner_preset: Option<String>,
    },

    /// Open an interactive preview window to debug the stitch.
    Preview {
        /// Input videos in calibration order (one panorama or two cameras).
        /// Chain segments within an input with `;`, as for `stitch`.
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<String>,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Window width in pixels.
        #[arg(long, default_value_t = 1280)]
        width: u32,

        /// Window height in pixels.
        #[arg(long, default_value_t = 720)]
        height: u32,

        /// Frame offset to sync left/right videos.
        /// Positive: skip N right frames (right started first).
        /// Negative: skip N left frames (left started first).
        #[arg(long, default_value_t = 0, allow_hyphen_values = true)]
        sync_offset: i64,

        /// Seam blend width (0.0 = hard cut). Overrides the calibration's
        /// saved value; when omitted, the calibration decides.
        #[arg(long, value_parser = parse_blend)]
        blend: Option<f32>,

        /// Rig tilt in degrees. Rotates the entire scene to compensate for
        /// a tilted camera rig, straightening vertical lines at the edges.
        #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
        rig_tilt: f32,
    },

    /// Stitch live camera feeds in real time.
    #[cfg(feature = "gstreamer")]
    Camera {
        /// Left camera device (sensor ID on Jetson, e.g. "0"; device path on Linux, e.g. "/dev/video0").
        #[arg(long)]
        left_device: String,

        /// Right camera device.
        #[arg(long)]
        right_device: String,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long)]
        output: String,

        /// Capture width in pixels.
        #[arg(long, default_value_t = 3840)]
        capture_width: u32,

        /// Capture height in pixels.
        #[arg(long, default_value_t = 2160)]
        capture_height: u32,

        /// Capture frame rate.
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
        capture_fps: u32,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Force a specific encoder (e.g. "libx264", "h264_nvenc").
        #[arg(long)]
        encoder: Option<String>,

        /// VAAPI render node or NVENC GPU index. Requires --encoder.
        #[arg(long, requires = "encoder")]
        encoder_device: Option<String>,

        /// Output codec: h264, hevc, av1. Defaults to the named encoder's codec, or h264.
        #[arg(long)]
        codec: Option<String>,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "fast")]
        quality: String,

        /// Seam blend width (0.0-1.0). Overrides the calibration's saved
        /// value; when omitted, the calibration decides.
        #[arg(long, value_parser = parse_blend)]
        blend: Option<f32>,

        /// Maximum number of frames to capture.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Stop after this many seconds.
        #[arg(long)]
        end_time: Option<f64>,

        /// Path to a YOLO ONNX model for ball detection and auto-tracking.
        #[arg(long)]
        model: Option<String>,

        /// Detection interval: run detection every N frames (default: 1).
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Directory for periodic JPEG snapshots of the stitched output
        /// (live preview for the gameday web panel). Requires the
        /// `snapshot` build feature.
        #[cfg(feature = "snapshot")]
        #[arg(long)]
        snapshot_dir: Option<String>,

        /// Write a snapshot every N frames (default: 30). Only effective
        /// when --snapshot-dir is set.
        #[cfg(feature = "snapshot")]
        #[arg(long, default_value_t = 30)]
        snapshot_interval: u64,

        /// Lookahead buffer in seconds for the live camera path. The panner
        /// leads the play by this window and the loop centered-smooths the
        /// pose lag-free, the same as `stitch --lookahead`. Adds ~this much
        /// latency and retained image storage. Only engages with AI tracking
        /// (needs --model, non-sweep); unavailable with raw Bayer capture.
        /// 0 = off (immediate render).
        #[arg(long, default_value_t = 0.75)]
        lookahead: f64,

        /// Override retained lookahead image storage in bytes (not total process memory).
        #[arg(long)]
        lookahead_memory_bytes: Option<usize>,

        /// Encoder quality on a 0-100 scale (higher = better). Overrides the
        /// quality preset. See `stitch --help` for the full mapping table.
        #[arg(long = "quality-value")]
        quality_value: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264; p1-p7 for NVENC).
        #[arg(long)]
        preset: Option<String>,

        /// Output container format. One of: `mp4` (default, needs
        /// close-time finalize), `fmp4` (fragmented MP4, streamable
        /// mid-write), `mkv` (Matroska, crash-safe + streamable).
        /// Use `mkv` for live recording (survives mid-session kills).
        #[arg(long)]
        container: Option<String>,

        /// RTMP stream URL for simultaneous file + stream output.
        /// The encoder writes the same packets to both the local
        /// file and this URL in a single encode pass (zero extra
        /// CPU). A silent audio track is added automatically for
        /// YouTube RTMP compatibility.
        #[arg(long)]
        stream_url: Option<String>,

        /// Tracking mode: `field` (ball + players, default), `ball`
        /// (ball only), `sweep` (no AI, slow pan; no --model needed).
        #[arg(long, default_value = "field", value_parser = ["field", "ball", "sweep"])]
        tracking: String,

        /// Disable the constrained-look coverage clamp. When off,
        /// the viewport can pan into black panorama edges — useful
        /// for sweeping the full coverage or debugging the coverage
        /// boundary itself.
        #[arg(long, default_value_t = false)]
        unconstrained: bool,

        /// Record pre-stitch source frames to this path as a
        /// stacked-video file. Same replay feature as
        /// `stitch --replay`. Requires `--features replay`.
        #[arg(long)]
        replay: Option<String>,

        /// Downscale replay tiles to `WIDTHxHEIGHT` (e.g.
        /// `1280x720`, `856x480`). GPU pack path only; width
        /// divisible by 4, height even. Requires `--replay`.
        #[arg(long, value_parser = parse_wxh)]
        replay_scale: Option<(u32, u32)>,

        /// Capture raw Bayer via V4L2 direct (bypasses NVIDIA ISP).
        /// Requires a custom camera driver and devices specified as
        /// V4L2 paths (e.g. `/dev/video0`). Runs GPU demosaic +
        /// ISP pipeline on the raw sensor data.
        #[arg(long, default_value_t = false)]
        v4l2_direct: bool,

        /// Sensor exposure time in microseconds (V4L2 direct only).
        /// Lower = darker. Outdoor daylight: 2000-8000. Indoor: 15000-30000.
        #[arg(long, default_value_t = 780)]
        exposure: u32,

        /// Sensor analog gain (V4L2 direct only, 16-357 for IMX477).
        /// Higher = brighter but noisier.
        #[arg(long, default_value_t = 16)]
        sensor_gain: u32,

        /// Run live calibration instead of stitching. Captures frame
        /// pairs from the cameras and aligns their images. Confirmed
        /// results are written to the --calibration path.
        #[arg(long, default_value_t = false)]
        live_calibrate: bool,

        /// Number of frame pairs to sample for live calibration.
        #[arg(long, default_value_t = 4)]
        calibrate_frames: usize,

        /// Left camera lens profile JSON (for calibration).
        /// Missing optics are estimated from captured images.
        #[arg(long)]
        left_lens_profile: Option<String>,

        /// Fixed right camera lens profile JSON; omitted optics are estimated.
        #[arg(long)]
        right_lens_profile: Option<String>,
    },

    /// Stitch live RPi CSI camera feeds via libcamera (rpicam-vid).
    #[cfg(feature = "libcamera")]
    Libcamera {
        /// Left camera index (e.g. 0).
        #[arg(long, default_value_t = 0)]
        left_camera: u32,

        /// Right camera index (e.g. 1).
        #[arg(long, default_value_t = 1)]
        right_camera: u32,

        /// Path to the calibration JSON file.
        #[arg(short, long)]
        calibration: String,

        /// Output file path.
        #[arg(short, long)]
        output: String,

        /// Capture width in pixels.
        #[arg(long, default_value_t = 1920)]
        capture_width: u32,

        /// Capture height in pixels.
        #[arg(long, default_value_t = 1080)]
        capture_height: u32,

        /// Capture frame rate.
        #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u32).range(1..))]
        capture_fps: u32,

        /// Output width in pixels.
        #[arg(long, default_value_t = 1920)]
        width: u32,

        /// Output height in pixels.
        #[arg(long, default_value_t = 1080)]
        height: u32,

        /// Force a specific encoder (e.g. "libx264", "h264_v4l2m2m").
        #[arg(long)]
        encoder: Option<String>,

        /// VAAPI render node or NVENC GPU index. Requires --encoder.
        #[arg(long, requires = "encoder")]
        encoder_device: Option<String>,

        /// Output codec: h264, hevc, av1. Defaults to the named encoder's codec, or h264.
        #[arg(long)]
        codec: Option<String>,

        /// Quality preset: fast, balanced, high.
        #[arg(long, default_value = "fast")]
        quality: String,

        /// Seam blend width (0.0-1.0). Overrides the calibration's saved
        /// value; when omitted, the calibration decides.
        #[arg(long, value_parser = parse_blend)]
        blend: Option<f32>,

        /// Maximum number of frames to capture.
        #[arg(long)]
        max_frames: Option<u64>,

        /// Stop after this many seconds.
        #[arg(long)]
        end_time: Option<f64>,

        /// Path to a YOLO model for ball detection and auto-tracking.
        #[arg(long)]
        model: Option<String>,

        /// Detection interval: run detection every N frames (default: 1).
        #[arg(long, default_value_t = 1)]
        detection_interval: u64,

        /// Encoder quality on a 0-100 scale (higher = better). Overrides the
        /// quality preset. See `stitch --help` for the full mapping table.
        #[arg(long = "quality-value")]
        quality_value: Option<u8>,

        /// Override encoder preset (e.g. ultrafast, veryfast, fast for x264).
        #[arg(long)]
        preset: Option<String>,
    },

    /// Align overlapping camera images and verify the fit on unused frames.
    Calibrate(calibrate::Args),

    /// Display information about the GPU and system capabilities.
    Info,

    /// Query a connected GoPro camera via USB or WiFi.
    #[cfg(feature = "gopro")]
    Gopro {
        /// GoPro serial number suffix (last 3 digits) for USB connection.
        /// Omit to use WiFi AP mode (10.5.5.9).
        #[arg(long)]
        serial: Option<String>,

        /// Connect via a custom URL instead of the default USB/WiFi address.
        #[arg(long)]
        url: Option<String>,

        /// Start recording on the camera.
        #[arg(long)]
        start: bool,

        /// Stop recording on the camera.
        #[arg(long)]
        stop: bool,

        /// Apply the sports stereo preset (disables HyperSmooth + horizon leveling).
        #[arg(long)]
        sports_preset: bool,
    },
}

/// Parse and validate blend width to [0.0, 1.0].
fn parse_blend(s: &str) -> Result<f32, String> {
    let v: f32 = s.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{v} is not in 0.0..=1.0"))
    }
}

/// Parse a `WIDTHxHEIGHT` string (e.g. `1280x720`, `854x480`) into
/// `(u32, u32)`. Used by `--replay-scale`. Validates YUV420P
/// alignment: width divisible by 4, height even.
fn parse_wxh(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {s:?}"))?;
    let w: u32 = w.parse().map_err(|e| format!("invalid width {w:?}: {e}"))?;
    let h: u32 = h
        .parse()
        .map_err(|e| format!("invalid height {h:?}: {e}"))?;
    if w == 0 || h == 0 {
        return Err(format!("dimensions must be > 0, got {w}x{h}"));
    }
    if !w.is_multiple_of(4) {
        return Err(format!(
            "width must be divisible by 4 (pack shader packs 4 pixels per u32 write), got {w}"
        ));
    }
    if !h.is_multiple_of(2) {
        return Err(format!(
            "height must be even (YUV420P chroma subsampling), got {h}"
        ));
    }
    Ok((w, h))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    #[cfg(target_os = "linux")]
    quiet_libva_info_by_default();

    // When profiling, tracing-subscriber is owned by the chrome layer
    // (one global subscriber per process). Otherwise, install our fmt
    // subscriber + log bridge for normal structured output.
    #[cfg(feature = "profiling")]
    let _profiling_guard = profiling::init()?;
    #[cfg(not(feature = "profiling"))]
    init_tracing();
    install_panic_hook();

    // Set up Ctrl-C handler so stitch can finalize the output file
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let interrupted = interrupted.clone();
        ctrlc::set_handler(move || {
            if interrupted.load(Ordering::Relaxed) {
                // Second Ctrl-C — force exit
                std::process::exit(1);
            }
            interrupted.store(true, Ordering::Relaxed);
            eprintln!("\nInterrupted — finishing output file...");
        })
        .expect("set Ctrl-C handler");
    }

    match cli.command {
        Commands::Stitch {
            inputs,
            live_input,
            audio_source,
            calibration,
            output,
            stream_url,
            width,
            height,
            start_time,
            end_time,
            max_frames,
            encoder,
            encoder_device,
            codec,
            quality,
            blend,
            panorama_arc,
            panorama_vertical_fov,
            sync_offset,
            model,
            detection_interval,
            lookahead,
            lookahead_memory_bytes,
            tracking,
            quality_value,
            preset,
            container,
            replay,
            replay_scale,
            no_zero_copy,
            cpu,
            events,
            trajectory,
            panner_config,
            panner_preset,
        } => stitch::run_stitch(
            stitch::StitchArgs {
                inputs: &inputs,
                live_input,
                audio_source,
                calibration: &calibration,
                output: &output,
                stream_url,
                width,
                height,
                blend,
                panorama_arc,
                panorama_vertical_fov,
                start_time,
                end_time,
                max_frames,
                encoder_name: encoder,
                encoder_device,
                codec: codec.as_deref(),
                quality: &quality,
                sync_offset,
                model_path: model.as_deref(),
                detection_interval,
                lookahead,
                lookahead_memory_bytes,
                tracking_mode: &tracking,
                quality_value,
                preset,
                container: container.as_deref(),
                replay_path: replay.as_deref(),
                replay_scale,
                no_zero_copy,
                cpu,
                events_path: events.as_deref(),
                trajectory_path: trajectory.as_deref(),
                panner_config_path: panner_config.as_deref(),
                panner_preset: panner_preset.as_deref(),
            },
            &interrupted,
        ),

        Commands::Preview {
            inputs,
            calibration,
            width,
            height,
            sync_offset,
            blend,
            rig_tilt,
        } => {
            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Preview dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );
            preview::run_preview(
                &preview::PreviewConfig {
                    inputs: &inputs,
                    calibration_path: &calibration,
                    width,
                    height,
                    sync_offset,
                    blend_width: blend,
                    rig_tilt_degrees: rig_tilt,
                },
                &interrupted,
            )
        }

        #[cfg(feature = "gstreamer")]
        Commands::Camera {
            left_device,
            right_device,
            calibration,
            output,
            capture_width,
            capture_height,
            capture_fps,
            width,
            height,
            encoder,
            encoder_device,
            codec,
            quality,
            blend,
            max_frames,
            end_time,
            model,
            detection_interval,
            #[cfg(feature = "snapshot")]
            snapshot_dir,
            #[cfg(feature = "snapshot")]
            snapshot_interval,
            lookahead,
            lookahead_memory_bytes,
            quality_value,
            preset,
            container,
            stream_url,
            tracking,
            unconstrained,
            replay,
            replay_scale,
            v4l2_direct,
            exposure,
            sensor_gain,
            live_calibrate,
            calibrate_frames,
            left_lens_profile,
            right_lens_profile,
        } => {
            use reco_io::gstreamer::camera::CameraConfig;

            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );

            log::info!("Camera capture: {left_device} + {right_device} -> {output}");

            let cam_config = CameraConfig {
                width: capture_width,
                height: capture_height,
                fps: capture_fps,
                left_device,
                right_device,
                synchronization: None,
            };

            if live_calibrate {
                return camera::run_live_calibrate(
                    cam_config,
                    &calibration,
                    capture_width,
                    capture_height,
                    calibrate_frames,
                    left_lens_profile.as_deref(),
                    right_lens_profile.as_deref(),
                    &interrupted,
                );
            }

            camera::run_camera(
                camera::CameraRunConfig {
                    cam_config,
                    capture_fps,
                    calibration: &calibration,
                    output: &output,
                    width,
                    height,
                    blend,
                    encoder_name: encoder,
                    encoder_device,
                    codec: codec.as_deref(),
                    quality: &quality,
                    end_time,
                    max_frames,
                    model_path: model.as_deref(),
                    detection_interval,
                    #[cfg(feature = "snapshot")]
                    snapshot_dir: snapshot_dir.as_deref(),
                    #[cfg(feature = "snapshot")]
                    snapshot_interval,
                    lookahead,
                    lookahead_memory_bytes,
                    quality_value,
                    preset,
                    container: container.as_deref(),
                    tracking: &tracking,
                    unconstrained,
                    replay_path: replay.as_deref(),
                    replay_scale,
                    stream_url: stream_url.as_deref(),
                    use_nvmm: !v4l2_direct && helpers::is_tegra(),
                    v4l2_direct,
                    exposure,
                    sensor_gain,
                },
                &interrupted,
            )
        }

        #[cfg(feature = "libcamera")]
        Commands::Libcamera {
            left_camera,
            right_camera,
            calibration,
            output,
            capture_width,
            capture_height,
            capture_fps,
            width,
            height,
            encoder,
            encoder_device,
            codec,
            quality,
            blend,
            max_frames,
            end_time,
            model,
            detection_interval,
            quality_value,
            preset,
        } => {
            use reco_io::libcamera::LibcameraConfig;

            const MAX_DIM: u32 = 8192;
            anyhow::ensure!(
                width > 0 && width <= MAX_DIM && height > 0 && height <= MAX_DIM,
                "Output dimensions {}x{} out of range: width and height must be 1..={MAX_DIM}",
                width,
                height,
            );

            log::info!("libcamera capture: cam{left_camera} + cam{right_camera} -> {output}");

            let cam_config = LibcameraConfig {
                width: capture_width,
                height: capture_height,
                fps: capture_fps,
                left_camera,
                right_camera,
            };

            libcamera_cmd::run_libcamera(
                libcamera_cmd::LibcameraRunConfig {
                    cam_config,
                    calibration: &calibration,
                    output: &output,
                    width,
                    height,
                    blend,
                    encoder_name: encoder,
                    encoder_device,
                    codec: codec.as_deref(),
                    quality: &quality,
                    end_time,
                    max_frames,
                    model_path: model.as_deref(),
                    detection_interval,
                    quality_value,
                    preset,
                },
                &interrupted,
            )
        }

        Commands::Calibrate(args) => calibrate::run_calibrate(args, &interrupted),

        Commands::Info => {
            let gpu = reco_core::gpu::GpuContext::new_blocking()?;
            println!("GPU: {}", gpu.gpu_name());
            println!("Backend: {}", gpu.backend_name());
            println!("Driver: {}", gpu.driver_info());

            println!("\nH.264 encoders:");
            let encoders = reco_io::ffmpeg::encoder::available_h264_encoders();
            if encoders.is_empty() {
                println!("  (none found)");
            } else {
                for enc in &encoders {
                    let tag = if enc.is_hardware { "HW" } else { "SW" };
                    println!("  {} [{}] — {}", enc.name, tag, enc.description);
                }
            }
            Ok(())
        }
        #[cfg(feature = "gopro")]
        Commands::Gopro {
            serial,
            url,
            start,
            stop,
            sports_preset,
        } => {
            use reco_control::gopro::GoProCamera;

            let cam = if let Some(url) = url {
                GoProCamera::connect_url(&url)?
            } else if let Some(serial) = serial {
                GoProCamera::connect_usb(&serial)?
            } else {
                println!("Trying WiFi AP mode (10.5.5.9)...");
                GoProCamera::connect_wifi()?
            };

            if let Some(info) = cam.info() {
                println!(
                    "Model:    {}",
                    info.model_name.as_deref().unwrap_or("unknown")
                );
                println!(
                    "Firmware: {}",
                    info.firmware_version.as_deref().unwrap_or("unknown")
                );
                println!(
                    "Serial:   {}",
                    info.serial_number.as_deref().unwrap_or("unknown")
                );
                println!("AP SSID:  {}", info.ap_ssid.as_deref().unwrap_or("unknown"));
            }

            let status = cam.status()?;
            println!("\nStatus:");
            if let Some(pct) = status.battery_percent {
                println!("  Battery:   {}%", pct);
            }
            if let Some(enc) = status.encoding {
                println!("  Recording: {}", if enc { "yes" } else { "no" });
            }
            if let Some(hot) = status.overheating {
                println!("  Overheat:  {}", if hot { "YES" } else { "no" });
            }

            if sports_preset {
                println!("\nApplying sports stereo preset...");
                cam.apply_sports_preset(
                    reco_control::gopro::VideoResolution::Res1080p,
                    reco_control::gopro::Fps::Fps30,
                    reco_control::gopro::VideoLens::Linear,
                )?;
                println!("Done.");
            }

            if start {
                cam.start_recording()?;
                println!("Recording started.");
            }

            if stop {
                cam.stop_recording()?;
                println!("Recording stopped.");
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stitch_rejects_unknown_tracking_modes_before_startup() {
        let args = [
            "reco",
            "stitch",
            "left.mp4",
            "right.mp4",
            "-c",
            "match.json",
            "--tracking",
        ];
        for mode in ["field", "ball", "sweep"] {
            assert!(Cli::try_parse_from(args.into_iter().chain([mode])).is_ok());
        }
        assert!(Cli::try_parse_from(args.into_iter().chain(["bal"])).is_err());
    }

    #[cfg(feature = "gstreamer")]
    #[test]
    fn camera_rejects_unknown_tracking_modes_before_startup() {
        let args = [
            "reco",
            "camera",
            "--left-device",
            "0",
            "--right-device",
            "1",
            "-c",
            "match.json",
            "-o",
            "out.mp4",
            "--tracking",
        ];
        for mode in ["field", "ball", "sweep"] {
            assert!(Cli::try_parse_from(args.into_iter().chain([mode])).is_ok());
        }
        assert!(Cli::try_parse_from(args.into_iter().chain(["bal"])).is_err());
    }

    #[test]
    fn preview_accepts_calibrated_input_lists() {
        assert!(Cli::try_parse_from(["reco", "preview", "-c", "cal.json"]).is_err());
        for paths in [
            vec!["panorama.mp4"],
            vec!["left1.mp4;left2.mp4", "right.mp4"],
        ] {
            let parsed = Cli::try_parse_from(
                ["reco", "preview"]
                    .into_iter()
                    .chain(paths.iter().copied())
                    .chain(["-c", "cal.json"]),
            )
            .unwrap();
            let Commands::Preview { inputs, .. } = parsed.command else {
                panic!("wrong command")
            };
            assert_eq!(inputs, paths);
        }
    }

    #[test]
    fn encoder_device_requires_a_named_encoder() {
        assert!(
            Cli::try_parse_from([
                "reco",
                "stitch",
                "left.mp4",
                "right.mp4",
                "-c",
                "match.json",
                "--encoder-device",
                "/dev/dri/renderD129",
            ])
            .is_err()
        );
        let parsed = Cli::try_parse_from([
            "reco",
            "stitch",
            "left.mp4",
            "right.mp4",
            "-c",
            "match.json",
            "--encoder",
            "hevc_vaapi",
            "--encoder-device",
            "/dev/dri/renderD129",
        ])
        .unwrap();
        let Commands::Stitch {
            codec,
            encoder_device,
            ..
        } = parsed.command
        else {
            panic!("wrong command");
        };
        assert_eq!(
            codec, None,
            "an omitted codec must not conflict with the named encoder"
        );
        assert_eq!(encoder_device.as_deref(), Some("/dev/dri/renderD129"));
    }

    #[test]
    fn panorama_cli_requires_both_extents() {
        let base = [
            "reco",
            "stitch",
            "left.mp4",
            "right.mp4",
            "-c",
            "match.json",
        ];
        for flag in ["--panorama-arc", "--panorama-vertical-fov"] {
            assert!(Cli::try_parse_from(base.into_iter().chain([flag, "90"])).is_err());
        }
        let parsed = Cli::try_parse_from(base.into_iter().chain([
            "--panorama-arc",
            "240",
            "--panorama-vertical-fov",
            "110",
        ]))
        .unwrap();
        let Commands::Stitch {
            panorama_arc,
            panorama_vertical_fov,
            ..
        } = parsed.command
        else {
            panic!("wrong command");
        };
        assert_eq!(panorama_arc, Some(240.0));
        assert_eq!(panorama_vertical_fov, Some(110.0));
    }

    #[cfg(feature = "gstreamer")]
    #[test]
    fn camera_rejects_zero_capture_fps_before_startup() {
        let args = [
            "reco",
            "camera",
            "--left-device",
            "0",
            "--right-device",
            "1",
            "-c",
            "match.json",
            "-o",
            "out.mp4",
            "--capture-fps",
        ];
        assert!(Cli::try_parse_from(args.into_iter().chain(["0"])).is_err());
        assert!(Cli::try_parse_from(args.into_iter().chain(["1"])).is_ok());
    }

    #[cfg(feature = "libcamera")]
    #[test]
    fn libcamera_rejects_zero_capture_fps_before_startup() {
        let args = [
            "reco",
            "libcamera",
            "-c",
            "match.json",
            "-o",
            "out.mp4",
            "--capture-fps",
        ];
        assert!(Cli::try_parse_from(args.into_iter().chain(["0"])).is_err());
        assert!(Cli::try_parse_from(args.into_iter().chain(["1"])).is_ok());
    }
}
