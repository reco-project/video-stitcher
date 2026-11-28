//! Trait adapters: wrap backend-specific types into `reco_core` traits.
//!
//! These are thin wrappers that bridge the gap between backend APIs
//! (FFmpeg, GStreamer) and the `FrameSource`/`Encoder` traits defined
//! in `reco-core`. Backend code stays clean and trait-free; all trait
//! plumbing lives here.

#[cfg(feature = "ffmpeg")]
use reco_core::sink::{OutputFrame, OutputSink, PixelFormat, SinkError, SinkInput};
#[cfg(feature = "ffmpeg")]
use reco_core::source::{FrameSet, SourceError, SourceFrame, SourceInfo, Yuv420pData};

#[cfg(feature = "ffmpeg")]
use crate::ffmpeg;

// -- FFmpeg File Source --

/// Settings differences for one camera, relative to its first recording.
#[cfg(feature = "ffmpeg")]
#[derive(Default)]
pub struct RecordingInspection {
    /// Encoded dimensions of the first recording, if probing succeeded.
    pub first_size: Option<(u32, u32)>,
    /// Frame rate of the first recording as a numerator and denominator.
    pub first_rate: Option<(i32, i32)>,
    /// Human-readable inspection warnings, including cross-camera differences.
    pub warnings: Vec<String>,
    /// Diagnostic text for each recording, in input order. Empty entries have no reported issue.
    pub recording_warnings: Vec<String>,
    /// One flag per recording. Cross-camera differences mark the nonbaseline camera.
    pub different: Vec<bool>,
}

/// Inspect every selected recording independently of decoder initialization.
///
/// Compares encoded dimensions, frame rate and display-matrix rotation within each camera's chain. Rotation differences between cameras are allowed.
/// Different camera dimensions can be supported; differing frame rates are
/// not resampled by the frame-pairing source. No warning certifies lens identity.
#[cfg(feature = "ffmpeg")]
pub fn inspect_recordings(inputs: &[crate::stitch_job::InputPath]) -> Vec<RecordingInspection> {
    probe_recordings(inputs).1
}

/// Estimate complete frame sets after chaining and two-camera alignment. An unknown segment makes the total unknown; partial counts are not totals.
#[cfg(all(feature = "ffmpeg", feature = "gpu"))]
pub(crate) fn input_frame_count(
    inputs: &[crate::stitch_job::InputPath],
    sync_offset: i64,
) -> Option<u64> {
    frame_set_count(&probe_recordings(inputs).0, sync_offset)
}

#[cfg(feature = "ffmpeg")]
fn probe_recordings(
    inputs: &[crate::stitch_job::InputPath],
) -> (Vec<Option<Vec<u64>>>, Vec<RecordingInspection>) {
    let mut warnings = Vec::new();
    let mut cameras = Vec::new();
    let counts = inputs
        .iter()
        .map(|input| {
            let mut issues = RecordingInspection::default();
            let mut baseline = None;
            let counts = input
                .all_paths()
                .iter()
                .enumerate()
                .map(
                    |(index, path)| match ffmpeg::decoder::probe_recording(path) {
                        Ok(metadata) => {
                            if index == 0 {
                                issues.first_size = Some(metadata.size);
                                issues.first_rate = metadata.rate.map(|rate| (rate.0, rate.1));
                            }
                            let settings = (metadata.size, metadata.rate, metadata.rotation);
                            let first = *baseline.get_or_insert(settings);
                            issues.different.push(first != settings);
                            let mut warning = String::new();
                            if first != settings {
                                let mut differences = Vec::new();
                                if first.0 != settings.0 {
                                    differences.push(format!(
                                        "resolution {} × {} (expected {} × {})",
                                        settings.0.0, settings.0.1, first.0.0, first.0.1
                                    ));
                                }
                                if first.1 != settings.1 {
                                    differences.push(format!(
                                        "frame rate {} (expected {})",
                                        recording_rate(settings.1),
                                        recording_rate(first.1)
                                    ));
                                }
                                if first.2 != settings.2 {
                                    differences.push(format!(
                                        "rotation {}° (expected {}°)",
                                        settings.2, first.2
                                    ));
                                }
                                warning = format!(
                                    "{}: {}.",
                                    path.file_name()
                                        .unwrap_or(path.as_os_str())
                                        .to_string_lossy(),
                                    differences.join("; ")
                                );
                                issues.warnings.push(warning.clone());
                            }
                            issues.recording_warnings.push(warning);
                            metadata.frames
                        }
                        Err(error) => {
                            issues.different.push(true);
                            let warning = format!(
                                "{}: recording settings could not be checked: {error}",
                                path.display()
                            );
                            issues.warnings.push(warning.clone());
                            issues.recording_warnings.push(warning);
                            None
                        }
                    },
                )
                // Visit every file even after an unknown count.
                .collect::<Vec<_>>()
                .into_iter()
                .collect();
            warnings.push(issues);
            cameras.push(baseline);
            counts
        })
        .collect();
    if let Some(Some(first)) = cameras.first() {
        for (index, settings) in cameras.iter().enumerate().skip(1) {
            if let Some(settings) = settings {
                let mut differences = Vec::new();
                if first.0 != settings.0 {
                    differences.push(format!("Camera resolutions differ: {} × {} and {} × {}. Per-camera dimensions are supported; each camera needs its matching lens setup.", first.0.0, first.0.1, settings.0.0, settings.0.1));
                }
                if let (Some(left), Some(right)) = (first.1, settings.1)
                    && left != right
                {
                    differences.push(format!("Camera frame rates differ: {:.3} and {:.3} fps. Reco pairs frames without resampling. Different rates can drift; use the same frame rate.", left.0 as f64 / left.1 as f64, right.0 as f64 / right.1 as f64));
                }
                if !differences.is_empty() {
                    warnings[index].different.fill(true);
                    for warning in &mut warnings[index].recording_warnings {
                        if !warning.is_empty() {
                            warning.push_str("\n\n");
                        }
                        warning.push_str(&differences.join("\n\n"));
                    }
                }
                warnings[0].warnings.extend(differences.clone());
                warnings[index].warnings.extend(differences);
            }
        }
    }
    (counts, warnings)
}

#[cfg(feature = "ffmpeg")]
fn recording_rate(rate: Option<ffmpeg_next::Rational>) -> String {
    rate.map_or_else(
        || "unknown".into(),
        |rate| format!("{:.3} fps", f64::from(rate.0) / f64::from(rate.1)),
    )
}

#[cfg(feature = "ffmpeg")]
fn frame_set_count(segments: &[Option<Vec<u64>>], sync_offset: i64) -> Option<u64> {
    let mut shortest = None;
    for (camera, counts) in segments.iter().enumerate() {
        let mut total = counts
            .as_ref()?
            .iter()
            .try_fold(0u64, |sum, count| sum.checked_add(*count))?;
        if segments.len() == 2
            && ((sync_offset > 0 && camera == 1) || (sync_offset < 0 && camera == 0))
        {
            total = total.saturating_sub(sync_offset.unsigned_abs());
        }
        shortest = Some(shortest.map_or(total, |previous: u64| previous.min(total)));
    }
    shortest
}

/// Keep a common source clock only when every camera has usable, compatible timing.
#[cfg(feature = "ffmpeg")]
pub(crate) fn merge_frame_rate(
    info: &mut SourceInfo,
    rate: Option<ffmpeg_next::Rational>,
    path: &std::path::Path,
) -> Result<(), SourceError> {
    match (info.fps_rational, rate) {
        (Some(_), Some(rate)) => {
            let fps = rate.0 as f64 / rate.1 as f64;
            // Permit nominal-rate rounding (30000/1001 vs 30), not camera drift (25 vs 30).
            if (fps - info.fps).abs() > 0.5 {
                return Err(SourceError::Init {
                    path: path.display().to_string(),
                    reason: format!(
                        "frame rate mismatch: this input is {fps:.2} fps but the first input is {:.2} fps; record all cameras at the same frame rate",
                        info.fps
                    ),
                });
            }
        }
        _ => {
            info.fps = 0.0;
            info.fps_rational = None;
        }
    }
    Ok(())
}

/// File source backed by one FFmpeg decoder per camera.
///
/// Opens one video per camera (projection order) and delivers
/// synchronized YUV420P [`FrameSet`]s. Each decoder runs in its own
/// thread; a zip thread assembles the per-camera frames after applying
/// the temporal sync offset.
///
/// ## Sync Offset
///
/// When cameras don't start recording at the same instant, a frame
/// offset aligns them temporally (defined between exactly two cameras;
/// see [`Self::open_with_offset`]):
/// - Positive offset: skip N frames from camera 1 (it started first)
/// - Negative offset: skip N frames from camera 0 (it started first)
#[cfg(feature = "ffmpeg")]
pub struct FfmpegFileSource {
    stream: crate::decode_stream::Restartable<Vec<Yuv420pData>>,
    // Field order disconnects the stream before joining its coordinator.
    cleanup: crate::decoder_cleanup::DecoderTask,
    info: SourceInfo,
    decode_backend: ffmpeg::decoder::DecodeBackend,
    full_range: bool,
    matrix: reco_core::render::color::YuvMatrix,
    color_metadata: ffmpeg::ColorMetadata,
    /// GPU pixel format (NV12 8-bit or P010 10-bit).
    #[cfg(feature = "gpu")]
    pixel_format: reco_core::render::renderer::GpuPixelFormat,
    /// Per-camera rotation from stream metadata (0, 90, 180, 270),
    /// in projection order.
    rotations: Vec<i32>,
    inputs: Vec<crate::stitch_job::InputPath>,
    segment_frames: Vec<Option<Vec<u64>>>,
    sync_offset: i64,
    /// Software decode forced at open; seek respawns must reuse it.
    software_decode: bool,
    /// Current frame position (incremented on each next_frame).
    current_frame: u64,
    /// Frames before this position are discarded by subsequent reads.
    skip_until: u64,
    /// True once the decode pipeline has signaled end-of-stream.
    ///
    /// Set only by explicit EOF, not by a failed or disconnected worker.
    /// Reset by a seek that respawns the decode pipeline.
    exhausted: bool,
    terminal_error: Option<SourceError>,
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
        Self::open_inputs(
            &[
                crate::stitch_job::InputPath::Single(left_path.to_path_buf()),
                crate::stitch_job::InputPath::Single(right_path.to_path_buf()),
            ],
            sync_offset,
            false,
        )
    }

    /// Open from a per-camera `InputPath` list in projection order.
    ///
    /// Accepts `InputPath` to support chained multi-segment files
    /// (GoPro/DJI auto-split). Format, fps, and decode backend are
    /// probed from input 0's first segment. Rotation is probed per
    /// input. Each chain must keep the same colour description and range.
    /// The frame-count estimate uses the shortest input.
    pub fn open_inputs(
        inputs: &[crate::stitch_job::InputPath],
        sync_offset: i64,
        software_decode: bool,
    ) -> Result<Self, SourceError> {
        let [first, ..] = inputs else {
            return Err(SourceError::Init {
                path: "(none)".into(),
                reason: "at least one input video is required".into(),
            });
        };
        for input in inputs {
            reco_core::source::validate_input_path(input.first_path())?;
        }

        let probe_path = first.first_path();
        let open_probe =
            |input| ffmpeg::decoder::VideoDecoder::open_input_probe(input, software_decode);
        let probe = open_probe(first).map_err(|e| SourceError::Init {
            path: probe_path.display().to_string(),
            reason: format!("{e}"),
        })?;
        let fps_r = probe.frame_rate();
        let full_range = probe.is_full_range();
        let first_color = probe.color_metadata();
        let fps = probe.fps().unwrap_or(0.0);
        let (segment_frames, recording_warnings) = probe_recordings(inputs);
        for warning in recording_warnings
            .iter()
            .flat_map(|report| &report.warnings)
        {
            log::warn!("{warning}");
        }
        let mut info = SourceInfo {
            camera_sizes: vec![(probe.width(), probe.height())],
            fps,
            fps_rational: fps_r.map(|r| (r.0, r.1)),
            total_frames: frame_set_count(&segment_frames, sync_offset),
        };
        let decode_backend = probe.backend();
        #[cfg(feature = "gpu")]
        let pixel_format = probe.pixel_format();
        let first_rotation = probe.rotation();
        drop(probe);

        let mut rotations = vec![first_rotation];
        let mut camera_colors = vec![first_color];
        for input in &inputs[1..] {
            let d = open_probe(input).map_err(|e| SourceError::Init {
                path: input.first_path().display().to_string(),
                reason: e.to_string(),
            })?;
            if d.is_full_range() != full_range {
                return Err(SourceError::Init {
                    path: input.first_path().display().to_string(),
                    reason: "camera color ranges differ; mixed full-range and limited-range inputs are not supported".into(),
                });
            }
            merge_frame_rate(&mut info, d.frame_rate(), input.first_path())?;
            info.camera_sizes.push((d.width(), d.height()));
            rotations.push(d.rotation());
            camera_colors.push(d.color_metadata());
        }
        let (color_metadata, matrix) =
            ffmpeg::ColorMetadata::reconcile(&camera_colors).map_err(|reason| {
                SourceError::Init {
                    path: probe_path.display().to_string(),
                    reason,
                }
            })?;

        let inputs = inputs.to_vec();
        let decode_inputs = inputs.clone();
        let (stream, coordinator) = crate::decode_stream::Restartable::new(move |tx, stop| {
            Self::decode_pipeline(decode_inputs, sync_offset, None, software_decode, tx, stop)
        })?;

        Ok(Self {
            stream,
            cleanup: crate::decoder_cleanup::DecoderTask::new(coordinator),
            full_range,
            matrix,
            color_metadata,
            info,
            decode_backend,
            #[cfg(feature = "gpu")]
            pixel_format,
            rotations,
            inputs,
            segment_frames,
            sync_offset,
            software_decode,
            current_frame: 0,
            skip_until: 0,
            exhausted: false,
            terminal_error: None,
        })
    }

    /// The decode backend selected during probe (CUDA, VAAPI, or software).
    pub fn decode_backend(&self) -> ffmpeg::decoder::DecodeBackend {
        self.decode_backend
    }

    /// Reconciled source colourimetry metadata across all cameras.
    pub fn color_metadata(&self) -> ffmpeg::ColorMetadata {
        self.color_metadata
    }

    /// Per-recording frame-count estimates in input order, before synchronization.
    ///
    /// Uses the same cached metadata as the chain total. Returns `None` for an
    /// invalid input index or if any recording count is unknown. These estimates
    /// describe back-to-back recordings, not evidence of capture-time gaps.
    pub fn segment_frame_counts(&self, input: usize) -> Option<&[u64]> {
        self.segment_frames.get(input)?.as_deref()
    }

    /// Transfer the cleanup wait to an interactive consumer's lifetime scope.
    ///
    /// By default, dropping a source cancels and joins its decoder workers.
    /// With a live scope, dropping the source only cancels them; the scope
    /// waits for cleanup when it is dropped. Seeking still waits for the old
    /// decoder generation before delivering replacement frames.
    pub fn defer_cleanup(&mut self, cleanup: &crate::DecoderCleanup) {
        self.cleanup.defer_to(cleanup);
    }

    fn read_frame(&mut self, blocking: bool) -> Result<Option<SourceFrame>, SourceError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        if self.exhausted {
            return Ok(None);
        }
        loop {
            let result = if blocking {
                self.stream.receive()
            } else {
                match self.stream.try_receive() {
                    Ok(result) => result,
                    Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(None),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        Err(crate::decode_stream::disconnected())
                    }
                }
            };
            let frames = result.inspect_err(|error| self.terminal_error = Some(error.clone()))?;
            let Some(images) = frames else {
                self.exhausted = true;
                return Ok(None);
            };
            let index = self.current_frame;
            self.current_frame += 1;
            if index >= self.skip_until {
                return Ok(Some(SourceFrame::untimed(FrameSet::Yuv420p(
                    images.into_iter().map(std::sync::Arc::new).collect(),
                ))));
            }
            // Poll at most one frame set, including when it must be discarded.
            if !blocking {
                return Ok(None);
            }
        }
    }

    /// GPU pixel format for zero-copy shared textures.
    ///
    /// Returns `GpuPixelFormat::P010` for 10-bit sources or
    /// `GpuPixelFormat::Nv12` for 8-bit.
    #[cfg(feature = "gpu")]
    pub fn pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        self.pixel_format
    }

    /// One camera's stream rotation from metadata (0, 90, 180, 270
    /// degrees), 0 for cameras this job does not have.
    pub fn rotation(&self, camera: reco_core::geometry::CameraIndex) -> i32 {
        self.rotations.get(camera).copied().unwrap_or(0)
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

    fn spawn_single_decoder_at(
        input: crate::stitch_job::InputPath,
        label: String,
        seek_secs: Option<f64>,
        software_decode: bool,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> (
        std::sync::mpsc::Receiver<crate::decode_stream::Message<Yuv420pData>>,
        std::thread::JoinHandle<()>,
    ) {
        crate::decode_stream::spawn(
            format!("decode_{label}"),
            4,
            move |tx| -> Result<(), ffmpeg::decoder::DecodeError> {
                use std::sync::atomic::Ordering;
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let open = if software_decode {
                    ffmpeg::decoder::VideoDecoder::open_input_software
                } else {
                    ffmpeg::decoder::VideoDecoder::open_input
                };
                let mut dec = open(&input)?;
                if stop.load(Ordering::Relaxed) {
                    return Ok(());
                }
                log::info!(
                    "{label} decoder: {} ({}x{})",
                    dec.backend(),
                    dec.width(),
                    dec.height()
                );
                crate::decode_stream::DecodeStream {
                    label: &format!("{label} decoder"),
                    start_secs: seek_secs,
                    stop: &stop,
                    tx,
                }
                .run(
                    &mut dec,
                    ffmpeg::decoder::VideoDecoder::next_frame,
                    |f| f.timestamp_us,
                    |f| Yuv420pData {
                        size: (f.width, f.height),
                        y: f.y,
                        u: f.u,
                        v: f.v,
                    },
                )
            },
        )
    }

    fn decode_pipeline(
        inputs: Vec<crate::stitch_job::InputPath>,
        sync_offset: i64,
        seek_secs: Option<f64>,
        software_decode: bool,
        tx: &std::sync::mpsc::SyncSender<crate::decode_stream::Message<Vec<Yuv420pData>>>,
        stop: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), SourceError> {
        use std::sync::atomic::Ordering;
        let mut decoders = FileDecoders {
            rxs: Vec::with_capacity(inputs.len()),
            workers: Vec::with_capacity(inputs.len()),
            stop: stop.clone(),
        };
        for (i, input) in inputs.into_iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let (rx, worker) = Self::spawn_single_decoder_at(
                input,
                format!("cam{i}"),
                seek_secs,
                software_decode,
                stop.clone(),
            );
            decoders.rxs.push(rx);
            decoders.workers.push(worker);
        }
        let rxs = &decoders.rxs;
        // Temporal alignment is defined between exactly two cameras.
        if let [cam0_rx, cam1_rx] = rxs.as_slice() {
            let skip_rx = if sync_offset > 0 { cam1_rx } else { cam0_rx };
            if !crate::decode_stream::skip(skip_rx, sync_offset.unsigned_abs())? {
                return Ok(());
            }
        } else if sync_offset != 0 {
            log::info!(
                "Sync offset {sync_offset} ignored: alignment is defined for two-camera jobs, this job has {}",
                rxs.len()
            );
        }

        while !stop.load(Ordering::Relaxed) {
            let mut cams = Vec::with_capacity(rxs.len());
            for rx in rxs {
                if let Some(frame) = crate::decode_stream::receive(rx)? {
                    cams.push(frame);
                }
            }
            // Read every camera's result before accepting EOF so a
            // failure at the same position as a shorter input isn't lost.
            if cams.len() != rxs.len() {
                break;
            }
            if tx.send(Ok(Some(cams))).is_err() {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "ffmpeg")]
struct FileDecoders {
    rxs: Vec<std::sync::mpsc::Receiver<crate::decode_stream::Message<Yuv420pData>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "ffmpeg")]
impl Drop for FileDecoders {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        // Release every blocked producer before waiting for any of them.
        self.rxs.clear();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                log::error!("File decoder panicked during cleanup");
            }
        }
    }
}

#[cfg(all(test, feature = "ffmpeg"))]
mod decode_tests {
    use super::*;

    #[test]
    fn segment_counts_form_the_same_total_and_keep_unknowns_unknown() {
        let counts = vec![Some(vec![10, 30, 20]), Some(vec![40, 30])];
        assert_eq!(frame_set_count(&counts, 0), Some(60));
        assert_eq!(frame_set_count(&counts, 20), Some(50));
        assert_eq!(frame_set_count(&counts, -20), Some(40));
        assert_eq!(frame_set_count(&[Some(vec![10]), None], 0), None);
        assert_eq!(frame_set_count(&[Some(vec![u64::MAX, 1])], 0), None);
        assert_eq!(frame_set_count(&[], 0), None);
    }

    #[test]
    fn recording_metadata_is_cached_and_unknown_inputs_are_not_invented() {
        let mut source = source(vec![]);
        source.segment_frames = vec![Some(vec![100, 300, 200]), None];
        assert_eq!(
            source.segment_frame_counts(0),
            Some([100, 300, 200].as_slice())
        );
        assert_eq!(source.segment_frame_counts(1), None);
        assert_eq!(source.segment_frame_counts(2), None);
    }
    use reco_core::source::FrameSource;

    fn source(messages: Vec<crate::decode_stream::Message<Vec<Yuv420pData>>>) -> FfmpegFileSource {
        let (tx, rx) = std::sync::mpsc::sync_channel(messages.len());
        for message in messages {
            tx.send(message).unwrap();
        }
        let stream = crate::decode_stream::Restartable::from_receiver(rx);
        FfmpegFileSource {
            cleanup: Default::default(),
            stream,
            info: SourceInfo {
                camera_sizes: vec![(2, 2); 2],
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            },
            decode_backend: ffmpeg::decoder::DecodeBackend::Software,
            full_range: false,
            matrix: Default::default(),
            color_metadata: ffmpeg::ColorMetadata::default(),
            #[cfg(feature = "gpu")]
            pixel_format: reco_core::render::renderer::GpuPixelFormat::Nv12,
            rotations: vec![0],
            inputs: vec![],
            segment_frames: vec![],
            sync_offset: 0,
            software_decode: true,
            current_frame: 0,
            skip_until: 0,
            exhausted: false,
            terminal_error: None,
        }
    }

    fn failure() -> crate::decode_stream::Message<Vec<Yuv420pData>> {
        Err(SourceError::Read {
            reason: "decode failed".into(),
        })
    }

    #[test]
    fn blocking_and_polling_reads_propagate_worker_failure() {
        for polling in [true, false] {
            let mut source = source(vec![Ok(Some(vec![])), failure()]);
            assert!(source.next_frame().unwrap().is_some());
            let error = if polling {
                source.try_next_frame()
            } else {
                source.next_frame()
            };
            assert!(error.err().unwrap().to_string().contains("decode failed"));
            assert!(
                source
                    .next_frame()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("decode failed")
            );
            assert!(
                source
                    .try_next_frame()
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("decode failed")
            );
            assert!(!source.is_exhausted());
            assert_eq!(source.current_frame, 1);
        }
    }

    #[test]
    fn forward_seek_does_not_swallow_a_worker_failure() {
        let mut source = source(vec![Ok(Some(vec![])), failure()]);
        source.seek(2).unwrap();
        assert_eq!(source.current_frame, 0, "request must not consume frames");
        assert!(
            source
                .next_frame()
                .err()
                .unwrap()
                .to_string()
                .contains("decode failed")
        );
        assert_eq!(source.current_frame, 1);
        assert!(!source.is_exhausted());
    }

    #[test]
    fn forward_seek_polling_is_bounded_and_newer_targets_replace_the_old_one() {
        let mut source = source((0..12).map(|_| Ok(Some(vec![]))).collect());
        source.seek(9).unwrap();
        assert!(source.try_next_frame().unwrap().is_none());
        assert_eq!(source.current_frame, 1);
        assert!(!source.is_exhausted());
        source.seek(2).unwrap();
        assert!(source.try_next_frame().unwrap().is_none());
        assert!(source.try_next_frame().unwrap().is_some());
        assert_eq!(source.current_frame, 3);
        assert!(source.next_frame().unwrap().is_some());
        assert_eq!(source.current_frame, 4);
    }

    #[test]
    fn forward_seek_preserves_eof_and_errors_in_both_read_modes() {
        for polling in [false, true] {
            for failed in [false, true] {
                let terminal = if failed { failure() } else { Ok(None) };
                let mut source = source(vec![Ok(Some(vec![])), terminal]);
                source.seek(3).unwrap();
                let result = if polling {
                    assert!(source.try_next_frame().unwrap().is_none());
                    assert!(!source.is_exhausted());
                    source.try_next_frame()
                } else {
                    source.next_frame()
                };
                assert_eq!(result.is_err(), failed);
                assert_eq!(source.is_exhausted(), !failed);
                assert_eq!(source.current_frame, 1);
                if !failed {
                    assert!(result.unwrap().is_none());
                }
            }
        }
    }

    #[test]
    fn unexpected_worker_disconnect_is_not_completion() {
        for polling in [true, false] {
            let mut source = source(vec![]);
            let result = if polling {
                source.try_next_frame()
            } else {
                source.next_frame()
            };
            assert!(result.err().unwrap().to_string().contains("disconnected"));
            assert!(!source.is_exhausted());
        }
    }

    #[test]
    fn worker_open_failure_reaches_the_receiver() {
        let dir = tempfile::tempdir().unwrap();
        let (rx, worker) = FfmpegFileSource::spawn_single_decoder_at(
            crate::InputPath::Single(dir.path().join("missing.mp4")),
            "test".into(),
            None,
            true,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        );
        assert!(crate::decode_stream::receive(&rx).is_err());
        worker.join().unwrap();
    }

    #[test]
    fn decoder_cleanup_disconnects_all_queues_and_waits_for_inflight_work() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, mpsc};
        use std::time::Duration;
        let stop = Arc::new(AtomicBool::new(false));
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (disconnected_tx, disconnected_rx) = mpsc::channel();
        let (rx, worker) = crate::decode_stream::spawn(
            "controlled_decode".into(),
            1,
            move |_| -> Result<(), SourceError> {
                entered_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(())
            },
        );
        let (queued_rx, queued_worker) = crate::decode_stream::spawn(
            "backpressured_decode".into(),
            1,
            move |tx| -> Result<(), SourceError> {
                while tx
                    .send(Ok(Some(Yuv420pData {
                        size: (2, 2),
                        y: vec![],
                        u: vec![],
                        v: vec![],
                    })))
                    .is_ok()
                {}
                disconnected_tx.send(()).unwrap();
                Ok(())
            },
        );
        let group = FileDecoders {
            rxs: vec![rx, queued_rx],
            workers: vec![worker, queued_worker],
            stop: stop.clone(),
        };
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let cleanup = std::thread::spawn(move || {
            drop(group);
            done_tx.send(()).unwrap();
        });
        // A native call may remain in flight even after cancellation. Cleanup
        // cannot finish or allow a replacement generation until it returns.
        disconnected_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        assert!(stop.load(Ordering::Relaxed));
        assert!(matches!(done_rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
        release_tx.send(()).unwrap();
        done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        cleanup.join().unwrap();
    }

    #[test]
    #[ignore = "requires RECO_DECODE_TEST_VIDEO pointing to a short, multi-frame video"]
    fn real_pipeline_drop_joins_decoders() {
        let path = std::path::PathBuf::from(std::env::var("RECO_DECODE_TEST_VIDEO").unwrap());
        for software in [true, false] {
            for offset in [-3, 0, 3] {
                let inputs = vec![crate::InputPath::Single(path.clone()); 2];
                let (stream, worker) = crate::decode_stream::Restartable::new(move |tx, stop| {
                    FfmpegFileSource::decode_pipeline(inputs, offset, Some(0.1), software, tx, stop)
                })
                .unwrap();
                assert!(stream.receive().unwrap().is_some());
                drop(stream);
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                let joiner = std::thread::spawn(move || done_tx.send(worker.join()).unwrap());
                done_rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap()
                    .unwrap();
                joiner.join().unwrap();
            }
        }
    }
}

#[cfg(feature = "ffmpeg")]
impl crate::stacked_video::replay::ReplaySource for FfmpegFileSource {
    fn replay_color_metadata(&self) -> crate::ffmpeg::ColorMetadata {
        self.color_metadata()
    }
}

#[cfg(feature = "ffmpeg")]
impl reco_core::source::FrameSource for FfmpegFileSource {
    fn yuv_matrix(&self) -> reco_core::render::color::YuvMatrix {
        self.matrix
    }

    fn is_full_range(&self) -> bool {
        self.full_range
    }

    fn info(&self) -> SourceInfo {
        self.info.clone()
    }

    fn left_rotation(&self) -> i32 {
        self.rotation(0)
    }

    fn right_rotation(&self) -> i32 {
        self.rotation(1)
    }

    #[cfg(feature = "gpu")]
    fn gpu_pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        self.pixel_format
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.read_frame(true)
    }

    fn try_next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        self.read_frame(false)
    }

    fn seek(&mut self, frame: u64) -> Result<(), SourceError> {
        if self.info.fps_rational.is_none() {
            return Err(SourceError::Read {
                reason: "video frame rate is unknown; cannot seek by frame index".into(),
            });
        }
        // Two strategies depending on seek direction and distance:
        //
        // 1. **Forward ≤ 10s**: reuse the existing decode pipeline. Reads
        //    discard intervening frames; polling bounds the work per call.
        //
        // 2. **Backward or > 10s forward**: replace the decode generation
        //    at the target position. Each decoder seeks to the
        //    nearest keyframe before the target, then decodes and
        //    discards frames until reaching the exact target PTS.
        //
        // Pending replacements coalesce behind cleanup of the old decoders.
        // Neither strategy waits for the target frame. A subsequent
        // blocking read waits for it. Native calls have no
        // shutdown deadline, so a stuck decoder also stalls the replacement.
        if frame >= self.current_frame && !self.exhausted && self.terminal_error.is_none() {
            let skip = frame - self.current_frame;
            let max_forward = (self.info.fps * 10.0) as u64;
            if skip <= max_forward {
                log::debug!("Forward seek: skipping {skip} frames via decode");
                self.skip_until = frame;
                return Ok(());
            }
        }

        let secs = frame as f64 / self.info.fps;
        log::debug!("Seeking to frame {frame} ({secs:.1}s)");
        let inputs = self.inputs.clone();
        let sync_offset = self.sync_offset;
        let software_decode = self.software_decode;
        self.stream.replace(move |tx, stop| {
            Self::decode_pipeline(inputs, sync_offset, Some(secs), software_decode, tx, stop)
        });
        self.current_frame = frame;
        self.skip_until = frame;
        self.exhausted = false;
        self.terminal_error = None;
        Ok(())
    }

    fn total_frames(&self) -> Option<u64> {
        self.info.total_frames
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
#[cfg(all(feature = "ffmpeg", feature = "gpu"))]
pub fn detect_zero_copy(source: &FfmpegFileSource, gpu: &reco_core::gpu::GpuContext) -> bool {
    std::env::var("RECO_NO_HWACCEL").is_err()
        && source.supports_zero_copy()
        && gpu.supports_zero_copy()
}

// -- FFmpeg File Encoder --

/// File encoder backed by FFmpeg.
///
/// Thin wrapper around `ffmpeg::encoder::VideoEncoder` that implements
/// the `reco_core::sink::OutputSink` trait.
#[cfg(feature = "ffmpeg")]
pub struct FfmpegFileEncoder {
    inner: ffmpeg::encoder::VideoEncoder,
}

#[cfg(feature = "ffmpeg")]
impl FfmpegFileEncoder {
    /// Encode limited-range rendered frames using the configured colour matrix.
    /// The sink requests NV12 for 8-bit output or P010LE for 10-bit output.
    ///
    /// Raw camera planes belong in [`ffmpeg::encoder::VideoEncoder`] instead.
    pub fn new(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps: (i32, i32),
        config: &ffmpeg::encoder::EncoderConfig,
    ) -> Result<Self, SinkError> {
        let matrix = config
            .color_metadata
            .matrix()
            .map_err(|reason| SinkError::Init { reason })?;
        let color = reco_core::render::color::YuvColor::delivery(matrix);
        Self::with_live_audio(path, width, height, fps, config, None, color)
    }

    pub(crate) fn with_live_audio(
        path: &std::path::Path,
        width: u32,
        height: u32,
        fps: (i32, i32),
        config: &ffmpeg::encoder::EncoderConfig,
        audio: Option<ffmpeg::live_audio::LiveAudio>,
        output_color: reco_core::render::color::YuvColor,
    ) -> Result<Self, SinkError> {
        let fps_rational = ffmpeg_next::Rational(fps.0, fps.1);
        let inner = ffmpeg::encoder::VideoEncoder::new_rendered(
            path,
            width,
            height,
            fps_rational,
            config,
            audio,
            output_color,
        )
        .map_err(|e| {
            if matches!(e, ffmpeg::encoder::EncodeError::Cancelled) {
                SinkError::Cancelled
            } else {
                SinkError::Init {
                    reason: e.to_string(),
                }
            }
        })?;
        Ok(Self { inner })
    }

    /// The name of the active encoder (e.g. "libx264", "h264_nvenc").
    pub fn encoder_name(&self) -> &str {
        self.inner.encoder_name()
    }

    /// Observe optional streaming while this sink is owned by a session worker, or after finalization.
    pub fn stream_monitor(&self) -> Option<ffmpeg::encoder::StreamMonitor> {
        self.inner.stream_monitor()
    }
}

#[cfg(feature = "ffmpeg")]
impl OutputSink for FfmpegFileEncoder {
    fn name(&self) -> &str {
        self.inner.encoder_name()
    }

    fn wants(&self) -> SinkInput {
        SinkInput::CpuBytes(self.inner.input_pixel_format())
    }

    fn consume(&mut self, frame: OutputFrame<'_>) -> Result<(), SinkError> {
        let result = match frame.format {
            PixelFormat::Nv12 => self.inner.write_nv12_frame(frame.data, frame.pts_us),
            PixelFormat::Rgba8 => self.inner.write_frame(frame.data, frame.pts_us),
            PixelFormat::P010 => self.inner.write_p010_frame(frame.data, frame.pts_us),
        };
        result.map_err(|e| match e {
            ffmpeg::encoder::EncodeError::Cancelled => SinkError::Cancelled,
            _ => SinkError::Consume {
                frame_index: None,
                reason: e.to_string(),
            },
        })
    }

    fn finish(&mut self) -> Result<(), SinkError> {
        self.inner.finish().map_err(|e| {
            if matches!(e, ffmpeg::encoder::EncodeError::Cancelled) {
                SinkError::Cancelled
            } else {
                SinkError::Finish {
                    reason: e.to_string(),
                }
            }
        })
    }
}
