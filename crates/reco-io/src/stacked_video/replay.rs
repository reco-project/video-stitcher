//! FrameSource decorator that records pre-stitch source frames to a
//! stacked-video file alongside the live pipeline.
//!
//! Enables the "professional replay" capability from the M6.5 plan
//! with zero plumbing in the consumer: call
//! [`crate::StitchJob::with_replay_recording`] and the library wires
//! a `ReplayRecordingSource` in front of the real source. Each
//! frame is passed through to the stitch pipeline unmodified while a
//! copy is packed into the replay encoder on the same thread.
//!
//! Only the CPU `StereoFrame::Yuv420p` variant is recorded today:
//!
//! - `Nv12` (Jetson ISP / NVDEC on Linux): not yet supported;
//!   would need an NV12 pack variant or a one-off Nv12->Yuv420p
//!   conversion on the hot path.
//! - `GpuResident` / `MetalResident`: can't record without a
//!   GPU->CPU readback per frame, which we don't pay until a
//!   consumer explicitly asks.
//!
//! A non-Yuv420p frame logs a `warn!` on first encounter and is
//! passed through unrecorded. The replay file will contain only the
//! Yuv420p-path frames.

use super::encoder::{StackedEncodeError, StackedEncoder, StackedEncoderConfig};
use super::{GridLayout, StackError};
use reco_core::core::types::StackedReplayGpuRecorder as CoreStackedGpuRecorder;
use reco_core::core::types::StackedReplayRecorder as CoreStackedReplayRecorder;
use reco_core::pipeline::YuvPlanes;
use reco_core::source::{FrameSource, SourceError, SourceInfo, StereoFrame, YuvFrame};
use reco_core::yuv_stack_packer::StackedAtlas;
use std::path::Path;

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
    /// the encoder parameters; `StackedEncoderConfig::default`
    /// is the usual choice.
    pub fn wrap(
        inner: Box<dyn FrameSource>,
        path: &Path,
        config: StackedEncoderConfig,
    ) -> Result<Self, StackedEncodeError> {
        let info = inner.info();
        let layout = GridLayout::vstack(info.width, info.height, 2).ok_or_else(|| {
            StackedEncodeError::Pack(StackError::TileDimensionMismatch {
                index: 0,
                got_w: info.width,
                got_h: info.height,
                expected_w: (info.width / 2) * 2,
                expected_h: (info.height / 2) * 2,
            })
        })?;
        let encoder = StackedEncoder::new(layout, path, config)?;
        log::info!(
            "Replay recording: {}x{} -> {} (stacked vertical, 2 tiles)",
            info.width,
            info.height,
            path.display(),
        );
        let _ = layout;
        Ok(Self {
            inner,
            encoder: Some(encoder),
            warned_non_yuv420p: false,
            frames_recorded: 0,
            // Flush to disk every ~1 second at 30fps. Balances
            // replay freshness against syscall overhead.
            flush_interval: 30,
        })
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

    fn record(&mut self, frame: &StereoFrame) {
        let Some(ref mut encoder) = self.encoder else {
            return;
        };
        match frame {
            StereoFrame::Yuv420p(pair) => {
                let info = self.inner.info();
                let left = YuvFrame {
                    y: pair.left.y.clone(),
                    u: pair.left.u.clone(),
                    v: pair.left.v.clone(),
                    width: info.width,
                    height: info.height,
                    timestamp_us: 0,
                };
                let right = YuvFrame {
                    y: pair.right.y.clone(),
                    u: pair.right.u.clone(),
                    v: pair.right.v.clone(),
                    width: info.width,
                    height: info.height,
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
                        "replay recording: source yields non-Yuv420p frames; \
                         recording disabled for this session (Nv12/GPU-resident \
                         variants need a separate pack path, not implemented yet)"
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

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        let frame = self.inner.next_frame()?;
        if let Some(ref f) = frame {
            self.record(f);
        }
        Ok(frame)
    }

    fn try_next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        let frame = self.inner.try_next_frame()?;
        if let Some(ref f) = frame {
            self.record(f);
        }
        Ok(frame)
    }

    fn is_gpu_resident(&self) -> bool {
        self.inner.is_gpu_resident()
    }

    fn gpu_pixel_format(&self) -> reco_core::renderer::GpuPixelFormat {
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
        // Best-effort finalize on drop so callers who forget
        // to call `finish()` still get a valid file (albeit
        // without the final-log line).
        let _ = self.finish();
    }
}

/// Compile-time bound: `ReplayRecordingSource` must be Send so
/// StitchJob can move it into the session's worker loop.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<ReplayRecordingSource>();
};

// -----------------------------------------------------------------
// Push-API recorder (FRICTION A18 close)
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
    /// new YuvFrame per call. Sized on first use then fixed.
    left_buf: Option<reco_core::source::YuvFrame>,
    right_buf: Option<reco_core::source::YuvFrame>,
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

    fn ensure_buf(buf: &mut Option<reco_core::source::YuvFrame>, width: u32, height: u32) {
        let needed_y = (width as usize) * (height as usize);
        let needed_uv = needed_y / 4;
        match buf {
            Some(f) if f.y.len() == needed_y && f.u.len() == needed_uv => {}
            _ => {
                *buf = Some(reco_core::source::YuvFrame {
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

    fn fill_from_planes(frame: &mut reco_core::source::YuvFrame, planes: &YuvPlanes<'_>) {
        // YuvPlanes is tight (no stride padding) by contract,
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
        left: &YuvPlanes<'_>,
        right: &YuvPlanes<'_>,
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

/// GPU-pack atlas recorder (M7 pivot). Implements
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
pub fn session_gpu_recorder(
    path: &std::path::Path,
    config: StackedEncoderConfig,
    atlas_width: u32,
    atlas_height: u32,
) -> Result<Box<dyn CoreStackedGpuRecorder>, StackedEncodeError> {
    GpuAtlasRecorder::open(path, config, atlas_width, atlas_height)
}
