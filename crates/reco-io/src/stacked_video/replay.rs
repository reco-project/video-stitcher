//! FrameSource decorator that records pre-stitch source frames to a
//! stacked-video file alongside the live pipeline.
//!
//! Enables replay-during-recording
//! with zero plumbing in the consumer: call
//! [`crate::StitchJob::with_replay_recording`] and the library wires
//! a `ReplayRecordingSource` in front of the real source. Each
//! frame is passed through to the stitch pipeline unmodified while a
//! copy is packed into the replay encoder on the same thread.
//!
//! `ReplayRecordingSource` records two-camera host YUV420P sets. Other frame formats are passed through unrecorded, with a warning on first encounter. With `gpu`, `GpuAtlasRecorder` accepts the core executor's packed GPU atlases; `StitchJob` chooses the recorder based on source residency.

use super::encoder::{StackedEncodeError, StackedEncoder, StackedEncoderConfig};
use super::{GridLayout, StackError};
use crate::ffmpeg::{ColorMetadata, encoder::EncodeError};
#[cfg(feature = "gpu")]
use reco_core::core::types::StackedReplayGpuRecorder as CoreStackedGpuRecorder;
use reco_core::core::types::StackedReplayRecorder as CoreStackedReplayRecorder;
#[cfg(feature = "gpu")]
use reco_core::gpu::yuv_stack_packer::StackedAtlas;
use reco_core::render::color::YuvMatrix;
use reco_core::render::planes::Yuv420pPlanes;
use reco_core::source::{
    FrameSet, FrameSource, SourceError, SourceFrame, SourceInfo, Yuv420pFrame,
};
use std::path::Path;

/// A frame source that exposes the stream tags needed for raw replay.
pub trait ReplaySource: FrameSource {
    /// Colourimetry tags carried by the source stream, if available.
    fn replay_color_metadata(&self) -> ColorMetadata {
        ColorMetadata::default()
    }
}

pub(crate) fn configure_replay_color(
    config: &mut StackedEncoderConfig,
    source: &dyn FrameSource,
    source_metadata: ColorMetadata,
) -> Result<(), EncodeError> {
    let matrix = source.yuv_matrix();
    let requested = config.inner.color_metadata.normalized();
    if requested.space.is_some() {
        let requested_matrix = requested.matrix().map_err(EncodeError::InvalidConfig)?;
        if requested_matrix != matrix {
            return Err(EncodeError::InvalidConfig(format!(
                "replay color matrix {requested_matrix:?} differs from source matrix {matrix:?}"
            )));
        }
    }

    config.full_range = source.is_full_range();
    config.inner.color_metadata = if matrix != YuvMatrix::Bt709 || requested.space.is_some() {
        source_metadata.with_matrix(matrix)
    } else {
        source_metadata
    };
    Ok(())
}

/// Decorator FrameSource that also writes source frames to a
/// stacked-video replay file.
pub struct ReplayRecordingSource {
    inner: Box<dyn FrameSource>,
    encoder: Option<StackedEncoder>,
    warned_non_yuv420p: bool,
    frames_recorded: u64,
    flush_interval: u64,
}

impl ReplayRecordingSource {
    /// Wrap an existing source, recording each Yuv420p stereo
    /// frame to `path` as it passes through. `config` supplies
    /// the encoder parameters. The recording uses the source range
    /// and colourimetry. An explicitly configured matrix must match.
    pub fn wrap(
        inner: Box<dyn ReplaySource>,
        path: &Path,
        mut config: StackedEncoderConfig,
    ) -> Result<Self, StackedEncodeError> {
        configure_replay_color(&mut config, inner.as_ref(), inner.replay_color_metadata())?;
        let encoder = Self::open_encoder(inner.as_ref(), path, config)?;
        Ok(Self::with_encoder(inner, encoder))
    }

    /// Open a recorder with the resolved source colour without consuming the source.
    pub(crate) fn open_encoder(
        inner: &dyn FrameSource,
        path: &Path,
        config: StackedEncoderConfig,
    ) -> Result<StackedEncoder, StackedEncodeError> {
        let info = inner.info();
        let [first, _] = info.camera_sizes.as_slice() else {
            return Err(StackedEncodeError::Encode(EncodeError::InvalidConfig(
                "stacked replay requires two cameras".into(),
            )));
        };
        let first = *first;
        for (index, &(got_w, got_h)) in info.camera_sizes.iter().enumerate() {
            if (got_w, got_h) != first {
                return Err(StackError::TileDimensionMismatch {
                    index,
                    got_w,
                    got_h,
                    expected_w: first.0,
                    expected_h: first.1,
                }
                .into());
            }
        }
        let layout = GridLayout::vstack(first.0, first.1, 2).ok_or_else(|| {
            StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                index: 0,
                got_w: first.0,
                got_h: first.1,
                expected_w: (first.0 / 2) * 2,
                expected_h: (first.1 / 2) * 2,
            })
        })?;
        let encoder = StackedEncoder::new(layout, path, config)?;
        log::info!(
            "Replay recording: {}x{} -> {} (stacked vertical, 2 tiles)",
            first.0,
            first.1,
            path.display(),
        );
        Ok(encoder)
    }

    /// Attach a recorder opened for this source with [`Self::open_encoder`].
    pub(crate) fn with_encoder(inner: Box<dyn FrameSource>, encoder: StackedEncoder) -> Self {
        Self {
            inner,
            encoder: Some(encoder),
            warned_non_yuv420p: false,
            frames_recorded: 0,
            // Flush to disk every ~1 second at 30fps. Balances
            // replay freshness against syscall overhead.
            flush_interval: 30,
        }
    }

    /// Finalize the replay file. Safe to call from the pipeline
    /// run-end path; subsequent frames from the inner source
    /// pass through unrecorded.
    pub fn finish(&mut self) -> Result<(), StackedEncodeError> {
        if let Some(mut enc) = self.encoder.take() {
            enc.finish()?;
            log::info!(
                "Replay recording: finished ({} frames)",
                self.frames_recorded
            );
        }
        Ok(())
    }

    /// How many frames have been written to the replay file.
    pub fn frames_recorded(&self) -> u64 {
        self.frames_recorded
    }

    fn record(&mut self, frame: &FrameSet) {
        let Some(ref mut encoder) = self.encoder else {
            return;
        };
        match frame {
            FrameSet::Yuv420p(cams) if cams.len() == 2 => {
                let left = Yuv420pFrame {
                    y: cams[0].y.clone(),
                    u: cams[0].u.clone(),
                    v: cams[0].v.clone(),
                    width: cams[0].size.0,
                    height: cams[0].size.1,
                    timestamp_us: 0,
                };
                let right = Yuv420pFrame {
                    y: cams[1].y.clone(),
                    u: cams[1].u.clone(),
                    v: cams[1].v.clone(),
                    width: cams[1].size.0,
                    height: cams[1].size.1,
                    timestamp_us: 0,
                };
                if let Err(e) = encoder.push(&[Some(&left), Some(&right)]) {
                    log::warn!(
                        "replay push failed ({e}); disabling replay recording for this session"
                    );
                    self.encoder = None;
                    return;
                }
                self.frames_recorded += 1;
                if self.frames_recorded.is_multiple_of(self.flush_interval) {
                    // Best-effort flush; on failure we keep
                    // encoding. The reader will see stale
                    // content but that's better than dropping
                    // the whole session.
                    let _ = encoder.flush();
                }
            }
            _ => {
                if !self.warned_non_yuv420p {
                    log::warn!(
                        "replay recording: source yields frames outside the 2-tile \
                         Yuv420p format; recording disabled for this session \
                         (Nv12/GPU-resident/mono sets need a separate pack path, \
                         not implemented yet)"
                    );
                    self.warned_non_yuv420p = true;
                    self.encoder = None;
                }
            }
        }
    }
}

impl FrameSource for ReplayRecordingSource {
    fn info(&self) -> SourceInfo {
        self.inner.info()
    }

    fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        let frame = self.inner.next_frame()?;
        if let Some(ref f) = frame {
            self.record(&f.images);
        }
        Ok(frame)
    }

    fn try_next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
        let frame = self.inner.try_next_frame()?;
        if let Some(ref f) = frame {
            self.record(&f.images);
        }
        Ok(frame)
    }

    fn is_gpu_resident(&self) -> bool {
        self.inner.is_gpu_resident()
    }

    #[cfg(feature = "gpu")]
    fn input_format(&self) -> reco_core::render::renderer::InputFormat {
        self.inner.input_format()
    }

    fn is_exhausted(&self) -> bool {
        self.inner.is_exhausted()
    }

    fn yuv_matrix(&self) -> reco_core::render::color::YuvMatrix {
        self.inner.yuv_matrix()
    }

    fn is_full_range(&self) -> bool {
        self.inner.is_full_range()
    }

    #[cfg(feature = "gpu")]
    fn gpu_pixel_format(&self) -> reco_core::render::renderer::GpuPixelFormat {
        self.inner.gpu_pixel_format()
    }

    fn left_rotation(&self) -> i32 {
        self.inner.left_rotation()
    }

    fn right_rotation(&self) -> i32 {
        self.inner.right_rotation()
    }
}

impl Drop for ReplayRecordingSource {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            log::warn!("replay recording finalize failed: {error}");
        }
    }
}

/// Compile-time bound: `ReplayRecordingSource` must be Send so
/// StitchJob can move it into the session's worker loop.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReplayRecordingSource>();
};

// -----------------------------------------------------------------
// Push-API recorder
// -----------------------------------------------------------------

/// Push-API stacked-video recorder. Implements reco-core's
/// [`reco_core::core::types::StackedReplayRecorder`] trait so any
/// [`reco_core::session::StitchSession`] can attach a recording
/// file with a single call.
///
/// Packs each submitted YUV plane pair into a [`GridLayout`] and
/// feeds the encoder. Uses the same [`StackedEncoder`] +
/// [`StackedEncoderConfig`] the pull side uses, so behavior,
/// container defaults, and GOP cadence are identical across the
/// two APIs.
pub struct SessionStackedRecorder {
    encoder: Option<StackedEncoder>,
    width: u32,
    height: u32,
    frames_recorded: u64,
    frames_failed: u64,
    /// Reusable buffers so `push_planes` avoids allocating a
    /// new Yuv420pFrame per call. Sized on first use then fixed.
    left_buf: Option<reco_core::source::Yuv420pFrame>,
    right_buf: Option<reco_core::source::Yuv420pFrame>,
}

impl SessionStackedRecorder {
    /// Open a recording for an N=2 vertical stack of
    /// `width x height` tiles. Returns a boxed trait object so
    /// consumers can pass it straight to
    /// [`reco_core::session::StitchSession::set_stacked_recorder`].
    pub fn open(
        path: &std::path::Path,
        config: StackedEncoderConfig,
        width: u32,
        height: u32,
    ) -> Result<Box<dyn CoreStackedReplayRecorder>, StackedEncodeError> {
        let layout =
            crate::stacked_video::GridLayout::vstack(width, height, 2).ok_or_else(|| {
                StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                    index: 0,
                    got_w: width,
                    got_h: height,
                    expected_w: (width / 2) * 2,
                    expected_h: (height / 2) * 2,
                })
            })?;
        let encoder = StackedEncoder::new(layout, path, config)?;
        log::info!(
            "SessionStackedRecorder: {}x{} tiles -> {} (push API, M6.5 A18)",
            width,
            height,
            path.display(),
        );
        Ok(Box::new(Self {
            encoder: Some(encoder),
            width,
            height,
            frames_recorded: 0,
            frames_failed: 0,
            left_buf: None,
            right_buf: None,
        }))
    }

    fn ensure_buf(buf: &mut Option<reco_core::source::Yuv420pFrame>, width: u32, height: u32) {
        let needed_y = (width as usize) * (height as usize);
        let needed_uv = needed_y / 4;
        match buf {
            Some(f) if f.y.len() == needed_y && f.u.len() == needed_uv => {}
            _ => {
                *buf = Some(reco_core::source::Yuv420pFrame {
                    y: vec![0u8; needed_y],
                    u: vec![0u8; needed_uv],
                    v: vec![0u8; needed_uv],
                    width,
                    height,
                    timestamp_us: 0,
                });
            }
        }
    }

    fn fill_from_planes(frame: &mut reco_core::source::Yuv420pFrame, planes: &Yuv420pPlanes<'_>) {
        // Yuv420pPlanes is tight (no stride padding) by contract,
        // so a straight copy_from_slice works. If the slices
        // are shorter than expected (malformed caller input)
        // we skip recording for this frame - the encoder would
        // reject a tile with mismatched plane sizes anyway.
        let need_y = frame.y.len();
        let need_uv = frame.u.len();
        if planes.y.len() < need_y || planes.u.len() < need_uv || planes.v.len() < need_uv {
            return;
        }
        frame.y.copy_from_slice(&planes.y[..need_y]);
        frame.u.copy_from_slice(&planes.u[..need_uv]);
        frame.v.copy_from_slice(&planes.v[..need_uv]);
    }
}

impl CoreStackedReplayRecorder for SessionStackedRecorder {
    fn record_yuv(
        &mut self,
        left: &Yuv420pPlanes<'_>,
        right: &Yuv420pPlanes<'_>,
        width: u32,
        height: u32,
    ) {
        // Reject dimension drift: the recorder was opened for
        // a fixed layout. If the session starts feeding
        // differently-sized frames (e.g. resolution change
        // mid-session) we stop recording and log.
        if width != self.width || height != self.height {
            log::warn!(
                "SessionStackedRecorder: frame {}x{} != opened {}x{}; disabling recording",
                width,
                height,
                self.width,
                self.height,
            );
            self.encoder = None;
            return;
        }
        let Some(ref mut encoder) = self.encoder else {
            return;
        };
        Self::ensure_buf(&mut self.left_buf, width, height);
        Self::ensure_buf(&mut self.right_buf, width, height);
        let (Some(ref mut left_frame), Some(ref mut right_frame)) =
            (self.left_buf.as_mut(), self.right_buf.as_mut())
        else {
            return;
        };
        Self::fill_from_planes(left_frame, left);
        Self::fill_from_planes(right_frame, right);
        match encoder.push(&[Some(left_frame), Some(right_frame)]) {
            Ok(()) => {
                self.frames_recorded += 1;
            }
            Err(e) => {
                self.frames_failed += 1;
                log::warn!(
                    "SessionStackedRecorder: push failed after {} frames ({e}); \
                     disabling recording",
                    self.frames_recorded,
                );
                self.encoder = None;
            }
        }
    }

    fn flush(&mut self) {
        if let Some(ref mut encoder) = self.encoder
            && let Err(e) = encoder.flush()
        {
            log::warn!("SessionStackedRecorder: flush failed ({e})");
        }
    }

    fn finish(&mut self) {
        if let Some(mut encoder) = self.encoder.take() {
            match encoder.finish() {
                Ok(()) => log::info!(
                    "SessionStackedRecorder: finished ({} frames recorded, {} failed)",
                    self.frames_recorded,
                    self.frames_failed,
                ),
                Err(e) => log::warn!(
                    "SessionStackedRecorder: finish failed ({e}) after {} frames",
                    self.frames_recorded,
                ),
            }
        }
    }
}

impl Drop for SessionStackedRecorder {
    fn drop(&mut self) {
        // Best-effort finalize so a recorder dropped without an
        // explicit `finish()` call still produces a valid file.
        if self.encoder.is_some() {
            self.finish();
        }
    }
}

/// Convenience constructor: opens a recorder suitable for
/// [`reco_core::session::StitchSession::set_stacked_recorder`].
/// Equivalent to [`SessionStackedRecorder::open`] but named for
/// discoverability from consumers looking at the session API
/// docs.
pub fn session_recorder(
    path: &std::path::Path,
    config: StackedEncoderConfig,
    width: u32,
    height: u32,
) -> Result<Box<dyn CoreStackedReplayRecorder>, StackedEncodeError> {
    SessionStackedRecorder::open(path, config, width, height)
}

/// GPU-pack atlas recorder. Implements
/// [`reco_core::core::types::StackedReplayGpuRecorder`] by forwarding
/// pre-packed [`StackedAtlas`] bytes straight to a
/// [`StackedEncoder`]. There's no pack work in this type - the
/// compute shader already produced the Y/U/V atlas planes; this
/// recorder is just the sink that reaches the encoder.
///
/// Open with [`GpuAtlasRecorder::open`], passing the
/// **atlas** dimensions (not per-tile dims) returned by
/// [`reco_core::session::StitchSession::stacked_atlas_dims`].
/// Attach via
/// [`reco_core::session::StitchSession::set_stacked_gpu_recorder`].
#[cfg(feature = "gpu")]
pub struct GpuAtlasRecorder {
    encoder: Option<StackedEncoder>,
    atlas_width: u32,
    atlas_height: u32,
    frames_recorded: u64,
    frames_failed: u64,
    /// Flush-to-disk cadence. Kept identical to
    /// [`SessionStackedRecorder`]: once per 30 atlases, which at
    /// 30 fps is roughly once per second.
    flush_interval: u64,
}

#[cfg(feature = "gpu")]
impl GpuAtlasRecorder {
    /// Open a GPU-pack atlas recorder. `atlas_width / atlas_height`
    /// are the full atlas dims the packer produces - for an N=2
    /// vstack of `tile_w x tile_h` tiles this is
    /// `(tile_w, tile_h * 2)`. The encoder is configured for a
    /// 1x1 layout of `atlas_width x atlas_height` since the pack
    /// shader already built the grid; there are no sub-tiles to
    /// pack here.
    pub fn open(
        path: &std::path::Path,
        config: StackedEncoderConfig,
        atlas_width: u32,
        atlas_height: u32,
    ) -> Result<Box<dyn CoreStackedGpuRecorder>, StackedEncodeError> {
        let layout = crate::stacked_video::GridLayout::vstack(atlas_width, atlas_height, 1)
            .ok_or_else(|| {
                StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                    index: 0,
                    got_w: atlas_width,
                    got_h: atlas_height,
                    expected_w: (atlas_width / 2) * 2,
                    expected_h: (atlas_height / 2) * 2,
                })
            })?;
        let encoder = StackedEncoder::new(layout, path, config)?;
        log::info!(
            "GpuAtlasRecorder: atlas {}x{} -> {} (GPU-pack sink, M7)",
            atlas_width,
            atlas_height,
            path.display(),
        );
        Ok(Box::new(Self {
            encoder: Some(encoder),
            atlas_width,
            atlas_height,
            frames_recorded: 0,
            frames_failed: 0,
            flush_interval: 30,
        }))
    }
}

#[cfg(feature = "gpu")]
impl CoreStackedGpuRecorder for GpuAtlasRecorder {
    fn record_atlas(&mut self, atlas: &StackedAtlas) {
        // Dimension drift: the packer was opened for a fixed
        // atlas size. Anything else indicates consumer wiring
        // confusion - stop recording, log once.
        if atlas.width != self.atlas_width || atlas.height != self.atlas_height {
            log::warn!(
                "GpuAtlasRecorder: atlas {}x{} != opened {}x{}; disabling recording",
                atlas.width,
                atlas.height,
                self.atlas_width,
                self.atlas_height,
            );
            self.encoder = None;
            return;
        }
        let Some(ref mut encoder) = self.encoder else {
            return;
        };
        match encoder.push_prepacked_yuv420p(&atlas.y, &atlas.u, &atlas.v) {
            Ok(()) => {
                self.frames_recorded += 1;
                if self.frames_recorded.is_multiple_of(self.flush_interval)
                    && let Err(e) = encoder.flush()
                {
                    log::warn!("GpuAtlasRecorder: flush failed ({e})");
                }
            }
            Err(e) => {
                self.frames_failed += 1;
                log::warn!(
                    "GpuAtlasRecorder: write failed after {} atlases ({e}); disabling",
                    self.frames_recorded,
                );
                self.encoder = None;
            }
        }
    }

    fn flush(&mut self) {
        if let Some(ref mut encoder) = self.encoder
            && let Err(e) = encoder.flush()
        {
            log::warn!("GpuAtlasRecorder: flush failed ({e})");
        }
    }

    fn finish(&mut self) {
        if let Some(mut encoder) = self.encoder.take() {
            match encoder.finish() {
                Ok(()) => log::info!(
                    "GpuAtlasRecorder: finished ({} atlases recorded, {} failed)",
                    self.frames_recorded,
                    self.frames_failed,
                ),
                Err(e) => log::warn!(
                    "GpuAtlasRecorder: finish failed ({e}) after {} atlases",
                    self.frames_recorded,
                ),
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl Drop for GpuAtlasRecorder {
    fn drop(&mut self) {
        if self.encoder.is_some() {
            <Self as CoreStackedGpuRecorder>::finish(self);
        }
    }
}

/// Convenience constructor for the GPU-pack recorder. Mirrors
/// [`session_recorder`] for the GPU path: the encoder type isn't
/// exposed to consumers; they receive a boxed trait object ready
/// for
/// [`reco_core::session::StitchSession::set_stacked_gpu_recorder`].
#[cfg(feature = "gpu")]
pub fn session_gpu_recorder(
    path: &std::path::Path,
    config: StackedEncoderConfig,
    atlas_width: u32,
    atlas_height: u32,
) -> Result<Box<dyn CoreStackedGpuRecorder>, StackedEncodeError> {
    GpuAtlasRecorder::open(path, config, atlas_width, atlas_height)
}

#[cfg(test)]
mod source_contract_tests {
    use super::*;

    struct PollingSource {
        polls: usize,
        full_range: bool,
        camera_sizes: Vec<(u32, u32)>,
    }

    impl FrameSource for PollingSource {
        fn info(&self) -> SourceInfo {
            SourceInfo {
                camera_sizes: self.camera_sizes.clone(),
                fps: 30.0,
                fps_rational: Some((30, 1)),
                total_frames: None,
            }
        }

        fn next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
            self.try_next_frame()
        }

        fn try_next_frame(&mut self) -> Result<Option<SourceFrame>, SourceError> {
            self.polls += 1;
            Ok(None)
        }

        fn is_exhausted(&self) -> bool {
            self.polls >= 2
        }

        fn is_full_range(&self) -> bool {
            self.full_range
        }
    }

    impl ReplaySource for PollingSource {}

    #[test]
    fn replay_wrapper_preserves_pending_eof_and_color_range() {
        for full_range in [false, true] {
            let mut source = ReplayRecordingSource {
                inner: Box::new(PollingSource {
                    polls: 0,
                    full_range,
                    camera_sizes: vec![(2, 2); 2],
                }),
                encoder: None,
                warned_non_yuv420p: false,
                frames_recorded: 0,
                flush_interval: 1,
            };
            assert_eq!(source.is_full_range(), full_range);
            assert!(!source.is_exhausted());
            assert!(source.try_next_frame().unwrap().is_none());
            assert!(!source.is_exhausted(), "pending is not EOF");
            assert!(source.try_next_frame().unwrap().is_none());
            assert!(source.is_exhausted());
            assert_eq!(source.is_full_range(), full_range);
        }
    }

    #[test]
    fn wrong_camera_count_fails_before_creating_replay_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replay.mkv");
        for count in [0, 1, 3] {
            let source = PollingSource {
                polls: 0,
                full_range: false,
                camera_sizes: vec![(128, 96); count],
            };
            let result = ReplayRecordingSource::wrap(
                Box::new(source),
                &path,
                StackedEncoderConfig::default(),
            );
            assert!(matches!(
                result,
                Err(StackedEncodeError::Encode(
                    crate::ffmpeg::encoder::EncodeError::InvalidConfig(_)
                ))
            ));
            assert!(!path.exists());
        }
    }

    #[test]
    fn unequal_camera_sizes_fail_before_creating_replay_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("replay.mkv");
        let source = PollingSource {
            polls: 0,
            full_range: false,
            camera_sizes: vec![(128, 96), (256, 192)],
        };
        let result =
            ReplayRecordingSource::wrap(Box::new(source), &path, StackedEncoderConfig::default());
        assert!(matches!(
            result,
            Err(StackedEncodeError::Pack(
                StackError::TileDimensionMismatch { index: 1, .. }
            ))
        ));
        assert!(!path.exists());
    }
}
