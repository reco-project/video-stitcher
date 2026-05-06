//! Configuration wiring methods for [`StitchSession`].
//!
//! Set/clear/attach methods that configure the session after
//! construction but before running. Each method is a thin setter
//! or delegates to the underlying [`StitchCore`](crate::core::StitchCore).

use super::StitchSession;
use crate::async_encode::AsyncEncodeThread;
use crate::encoder::Encoder;
use crate::session::types::{DetectionSink, ErrorPolicy};

impl StitchSession {
    /// Attach an encoder to this session.
    ///
    /// The encoder is moved to a background thread for async encoding.
    /// `buffer_count` controls how many frames can be in-flight between
    /// the render thread and the encode thread (typically 2).
    ///
    /// Must be called before [`Self::submit_render_output`], [`Self::process_frame`],
    /// or [`Self::run`].
    pub fn set_encoder(&mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) {
        let width = self.nv12_converter.width();
        let height = self.nv12_converter.height();
        self.encoder = Some(AsyncEncodeThread::new(encoder, width, height, buffer_count));
    }

    /// Add an additional encoder for multi-output (e.g. record + stream).
    ///
    /// The NV12 data from each rendered frame is fanned out to all attached
    /// encoders. Each encoder runs on its own background thread.
    ///
    /// Use [`set_encoder`](Self::set_encoder) for the primary encoder,
    /// then `add_encoder` for additional outputs.
    pub fn add_encoder(&mut self, encoder: Box<dyn Encoder + Send>, buffer_count: usize) {
        let width = self.nv12_converter.width();
        let height = self.nv12_converter.height();
        self.extra_encoders
            .push(AsyncEncodeThread::new(encoder, width, height, buffer_count));
    }

    /// Attach a [`UnifiedDetector`](crate::detector::UnifiedDetector)
    /// for object detection on raw camera frames.
    ///
    /// The backend declares which [`DetectorFrame`](crate::detector::DetectorFrame)
    /// residencies it accepts. Session dispatches CPU frames (YUV /
    /// NV12) and CUDA frames (shared textures) through the same
    /// detector; backends return `UnsupportedFrameKind` for residencies
    /// they cannot handle and session logs+drops those at the boundary.
    pub fn set_detector(&mut self, detector: Box<dyn crate::detector::UnifiedDetector>) {
        self.detection.set_detector(detector);
    }

    /// Set the detection interval (run detection every N frames).
    ///
    /// Default is 1 (every frame). Higher values reduce detection CPU load
    /// at the cost of tracking responsiveness. The director still receives
    /// the last known tracked objects on skipped frames.
    pub fn set_detection_interval(&mut self, interval: u64) {
        self.detection.set_detection_interval(interval);
    }

    /// Attach a pipeline event sink for structured observability.
    ///
    /// See [`crate::pipeline_event`] for the event vocabulary and the
    /// `BackpressuredSink` wrapper that keeps emission off the render
    /// thread. Typical usage:
    ///
    /// ```rust,ignore
    /// use reco_core::pipeline_event::BackpressuredSink;
    /// use reco_io::jsonl_sink::JsonlSink;
    ///
    /// let inner = JsonlSink::create("trace.jsonl")?;
    /// let sink = BackpressuredSink::new(Box::new(inner), 256, None);
    /// session.set_event_sink(Box::new(sink));
    /// ```
    ///
    /// Pass [`None`] equivalent by not calling this at
    /// all. There is deliberately no `clear_event_sink` - in a
    /// <1.0.0 codebase we re-create the session for that. When an
    /// external consumer hits this friction we'll add one.
    pub fn set_event_sink(&mut self, sink: Box<dyn crate::pipeline_event::PipelineEventSink>) {
        log::info!("StitchSession: event sink attached");
        self.event_sink = Some(sink);
    }

    /// Append a [`DetectionFilter`](crate::detection_filter::DetectionFilter)
    /// to the pre-tracker chain. Filters run in insertion order before
    /// the trackers see the detection list. With an event sink
    /// attached, each stage emits
    /// `PipelineEvent::DetectionFilter { before, after, filter_name }`.
    ///
    /// Typical chain:
    /// 1. `FlickerFilter` (recurrent static false-positive rejection).
    /// 2. Class-specific filters (feet-in-ROI, hands-raised, etc).
    pub fn add_detection_filter(
        &mut self,
        filter: Box<dyn crate::detection_filter::DetectionFilter>,
    ) {
        log::info!("StitchSession: detection filter '{}' added", filter.name());
        self.detection_filters.push(filter);
    }

    /// Attach a singleton ball tracker. See
    /// [`StitchCore::set_ball_tracker`](crate::core::StitchCore::set_ball_tracker)
    /// for semantics - the session mirrors the core's API so push
    /// and pull consumers stay symmetric.
    pub fn set_ball_tracker(&mut self, tracker: Box<dyn crate::tracker::Tracker>) {
        log::info!(
            "StitchSession: ball tracker attached (class_id={})",
            tracker.class_id()
        );
        self.ball_tracker = Some(tracker);
    }

    /// Remove the currently attached ball tracker.
    pub fn clear_ball_tracker(&mut self) {
        self.ball_tracker = None;
    }

    /// Attach a multi-entity player tracker. Mirror of
    /// [`StitchCore::set_player_tracker`](crate::core::StitchCore::set_player_tracker).
    pub fn set_player_tracker(&mut self, tracker: Box<dyn crate::tracker::Tracker>) {
        log::info!(
            "StitchSession: player tracker attached (class_id={})",
            tracker.class_id()
        );
        self.player_tracker = Some(tracker);
    }

    /// Remove the currently attached player tracker.
    pub fn clear_player_tracker(&mut self) {
        self.player_tracker = None;
    }

    /// Attach a panner. When set, the tracker/panner path owns
    /// pose resolution each frame; without a panner the pose stays at
    /// the pipeline default.
    pub fn set_panner(&mut self, panner: Box<dyn crate::panner::Panner>) {
        log::info!("StitchSession: panner attached");
        self.panner = Some(panner);
    }

    /// Remove the currently attached panner.
    pub fn clear_panner(&mut self) {
        log::info!("StitchSession: panner detached");
        self.panner = None;
    }

    /// Attach a stacked-video replay recorder.
    ///
    /// Forwards to `StitchCore::set_stacked_recorder` on the
    /// session's underlying core. Push-based consumers (OBS,
    /// GStreamer bridge) that wire this get the same replay-recording
    /// ergonomics the pull-side `StitchJob::with_replay_recording`
    /// already provides: one method call, the session handles the
    /// per-frame tap + encoder lifecycle internally. Closes FRICTION
    /// A18 on the reco-obs side.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // reco-io exposes a constructor that returns the concrete
    /// // `Box<dyn StackedReplayRecorder>`; consumers don't touch
    /// // the encoder type directly.
    /// let recorder = reco_io::stacked_video::replay::session_recorder(
    ///     "replay.mkv",
    ///     reco_io::stacked_video::encoder::StackedEncoderConfig::default(),
    ///     info.width,
    ///     info.height,
    /// )?;
    /// session.set_stacked_recorder(recorder);
    /// ```
    pub fn set_stacked_recorder(
        &mut self,
        recorder: Box<dyn crate::core::types::StackedReplayRecorder>,
    ) {
        self.core.set_stacked_recorder(recorder);
    }

    /// Finalize and drop the currently attached replay recorder.
    /// No-op if no recorder is attached.
    pub fn clear_stacked_recorder(&mut self) {
        self.core.clear_stacked_recorder();
    }

    /// Flush the replay recorder's buffered bytes to disk. Call
    /// periodically (e.g. once per second) so a concurrent reader
    /// sees recent frames. No-op if no recorder is attached.
    pub fn flush_stacked_recorder(&mut self) {
        self.core.flush_stacked_recorder();
    }

    /// Enable the GPU-pack replay path (M7 pivot item 1).
    ///
    /// Forwards to [`crate::core::StitchCore::enable_gpu_stacked_replay`].
    /// After enabling, attach a
    /// [`crate::core::types::StackedReplayGpuRecorder`] via
    /// [`Self::set_stacked_gpu_recorder`] to route the packed atlas
    /// to an encoder. The pack runs on every YUV submit and logs
    /// its path choice once at enable time.
    pub fn enable_gpu_stacked_replay(
        &mut self,
        layout: crate::yuv_stack_packer::StackGridLayout,
        output_size: crate::yuv_stack_packer::OutputTileSize,
    ) -> Result<(), crate::core::types::StitchCoreError> {
        self.core.enable_gpu_stacked_replay(layout, output_size)
    }

    /// Disable the GPU-pack replay path. Also finalizes any
    /// attached GPU recorder.
    pub fn disable_gpu_stacked_replay(&mut self) {
        self.core.disable_gpu_stacked_replay();
    }

    /// Attach a GPU-pack atlas recorder. Call after
    /// [`Self::enable_gpu_stacked_replay`].
    pub fn set_stacked_gpu_recorder(
        &mut self,
        recorder: Box<dyn crate::core::types::StackedReplayGpuRecorder>,
    ) {
        self.core.set_stacked_gpu_recorder(recorder);
    }

    /// Finalize and drop the GPU-pack atlas recorder. No-op if none
    /// is attached.
    pub fn clear_stacked_gpu_recorder(&mut self) {
        self.core.clear_stacked_gpu_recorder();
    }

    /// Flush the GPU-pack recorder's buffered bytes to disk. No-op
    /// if none is attached.
    pub fn flush_stacked_gpu_recorder(&mut self) {
        self.core.flush_stacked_gpu_recorder();
    }

    /// Atlas dimensions the active GPU packer produces, or `None` if
    /// the GPU-pack path is not enabled. Consumers use this to open
    /// an encoder sized for the atlas.
    pub fn stacked_atlas_dims(&self) -> Option<(u32, u32)> {
        self.core.stacked_atlas_dims()
    }

    /// Set the sink that receives tracked detection data each frame.
    ///
    /// The sink is called once per frame with the current tracked
    /// objects, frame index, and timestamp. Errors returned from the
    /// sink abort the current session call ([`run`](Self::run),
    /// [`step`](Self::step), [`process_frame`](Self::process_frame))
    /// with [`SessionError::DetectionSink`](super::types::SessionError::DetectionSink).
    ///
    /// Closures matching `FnMut(&[MappedDetection], u64, f64) -> Result<(),
    /// DetectionSinkError>` implement [`DetectionSink`] automatically via
    /// the blanket impl, so typical usage is:
    ///
    /// ```rust,ignore
    /// session.set_detection_sink(Box::new(|dets, frame_idx, ts_ms| {
    ///     writer.write_row(dets, frame_idx, ts_ms)?;
    ///     Ok(())
    /// }));
    /// ```
    ///
    /// Replaces any previously registered sink.
    pub fn set_detection_sink(&mut self, sink: Box<dyn DetectionSink>) {
        self.detection.set_sink(sink);
    }

    /// Set the error policy for the [`run()`](Self::run) batch loop.
    pub fn set_error_policy(&mut self, policy: ErrorPolicy) {
        self.error_policy = policy;
    }

    /// Update calibration parameters and recompute coverage boundary.
    ///
    /// Takes effect on the next render call. For interactive calibration
    /// tweaking during preview or live operation. Delegates to
    /// `StitchCore::update_calibration` which re-derives the coverage
    /// boundary in one call.
    pub fn update_calibration(&mut self, calibration: crate::calibration::MatchCalibration) {
        self.core.update_calibration(calibration);
    }
}
