//! Stacked replay wiring for [`StitchCore`](super::StitchCore).
//!
//! All methods related to the stacked replay recording system: CPU-path
//! ([`StackedReplayRecorder`](super::types::StackedReplayRecorder)) and
//! GPU-pack path ([`YuvStackPacker`](crate::yuv_stack_packer::YuvStackPacker)
//! + [`StackedReplayGpuRecorder`](super::types::StackedReplayGpuRecorder)).

use super::types::{StackedReplayGpuRecorder, StackedReplayRecorder, StitchCoreError};
use crate::renderer::InputFormat;
use crate::yuv_stack_packer::{
    OutputTileSize, SourceFormat, StackGridLayout, StackedPackSource, YuvStackPacker,
};

impl super::StitchCore {
    /// Attach a stacked-video replay recorder (M6.5 item 3, push
    /// side).
    ///
    /// Every subsequent YUV submit feeds the recorder before
    /// rendering. Errors inside the recorder are swallowed so a
    /// failing recorder cannot break the live stitch output; the
    /// recorder's own implementation is expected to log any
    /// failure. See [`StackedReplayRecorder`] for the full
    /// contract.
    ///
    /// Dropping an existing recorder via [`Self::clear_stacked_recorder`]
    /// is required before attaching a new one; otherwise the old
    /// recording is quietly abandoned.
    pub fn set_stacked_recorder(&mut self, recorder: Box<dyn StackedReplayRecorder>) {
        if self.stacked_recorder.is_some() {
            log::warn!(
                "StitchCore::set_stacked_recorder replacing an existing recorder; \
                 call clear_stacked_recorder first to finalize the previous recording"
            );
        }
        log::info!("StitchCore: stacked-video replay recorder attached");
        self.stacked_recorder = Some(recorder);
    }

    /// Drop the current replay recorder, calling `finish` first so
    /// the recording file is finalized. No-op if no recorder is
    /// attached.
    pub fn clear_stacked_recorder(&mut self) {
        if let Some(mut recorder) = self.stacked_recorder.take() {
            recorder.finish();
            log::info!("StitchCore: stacked-video replay recorder detached");
        }
    }

    /// Flush the replay recorder's buffered bytes to disk. Call
    /// periodically (e.g. once per second from a timer) so a
    /// concurrent reader sees recent frames. No-op if no recorder
    /// is attached.
    pub fn flush_stacked_recorder(&mut self) {
        if let Some(ref mut recorder) = self.stacked_recorder {
            recorder.flush();
        }
    }

    /// Enable the GPU-pack replay path (M7 pivot item 1).
    ///
    /// Builds a [`YuvStackPacker`] sized for `layout` x `output_size`
    /// and wires it into subsequent YUV submit calls. The packer's
    /// source-format variant is derived from the pipeline's input
    /// format so consumers don't risk a YUV/NV12 mismatch.
    ///
    /// Call [`Self::set_stacked_gpu_recorder`] to attach an
    /// atlas-consuming sink (typically a
    /// [`reco_io`](../../../reco_io/index.html) encoder) before the
    /// first submit, or later - the pack still runs either way and
    /// the first two submits are warmup.
    ///
    /// Emits one `log::info!` line stating the pack path has been
    /// chosen (GPU), the tile dims, `N`, and the source format - so
    /// the CPU vs GPU decision is never silent.
    ///
    /// Returns `StitchCoreError::Config` when the pipeline's input
    /// format is BGRA (the pack shader only handles YUV420P / NV12)
    /// or when the layout / output dims violate YUV420P alignment.
    pub fn enable_gpu_stacked_replay(
        &mut self,
        layout: StackGridLayout,
        output_size: OutputTileSize,
    ) -> Result<(), StitchCoreError> {
        let source_format = match self.pipeline.input_format() {
            InputFormat::Yuv420p => SourceFormat::Yuv420p,
            InputFormat::Nv12 => SourceFormat::Nv12,
            InputFormat::Bgra => {
                return Err(StitchCoreError::Config(
                    "GPU stacked replay requires YUV420P or NV12 input; BGRA pipelines must use \
                     the CPU replay-recording path"
                        .into(),
                ));
            }
        };
        let packer = YuvStackPacker::new(self.pipeline.gpu(), layout, output_size, source_format)?;
        let (atlas_w, atlas_h) = packer.atlas_dims();
        log::info!(
            "reco-core: replay pack path = GPU shader (tiles {}x{} out, N={}, atlas {}x{}, source_format={:?})",
            output_size.width,
            output_size.height,
            layout.capacity(),
            atlas_w,
            atlas_h,
            source_format,
        );
        if self.stacked_recorder.is_some() {
            log::warn!(
                "StitchCore::enable_gpu_stacked_replay: a CPU StackedReplayRecorder is also \
                 attached; both paths will run and duplicate work. Clear one to avoid \
                 redundant recording."
            );
        }
        self.stacked_packer = Some(packer);
        Ok(())
    }

    /// Disable the GPU-pack replay path and drop the packer.
    /// Also calls `finish` on any attached GPU recorder so its file
    /// is finalized. No-op when the path was not enabled.
    pub fn disable_gpu_stacked_replay(&mut self) {
        if self.stacked_packer.take().is_some() {
            log::info!("StitchCore: GPU stacked replay disabled");
        }
        self.clear_stacked_gpu_recorder();
    }

    /// Attach a GPU-pack atlas recorder. Must be called after
    /// [`Self::enable_gpu_stacked_replay`] for the pack output to
    /// reach disk - without a recorder the packer still runs but
    /// the readback bytes are dropped.
    pub fn set_stacked_gpu_recorder(&mut self, recorder: Box<dyn StackedReplayGpuRecorder>) {
        if self.stacked_gpu_recorder.is_some() {
            log::warn!(
                "StitchCore::set_stacked_gpu_recorder replacing an existing GPU recorder; \
                 call clear_stacked_gpu_recorder first to finalize the previous recording"
            );
        }
        if self.stacked_packer.is_none() {
            log::warn!(
                "StitchCore::set_stacked_gpu_recorder called before \
                 enable_gpu_stacked_replay: recorder will receive no atlases until the \
                 packer is enabled"
            );
        }
        log::info!("StitchCore: GPU stacked-replay recorder attached");
        self.stacked_gpu_recorder = Some(recorder);
    }

    /// Drop the GPU-pack atlas recorder, calling `finish` so the
    /// output file is finalized. No-op if no recorder is attached.
    pub fn clear_stacked_gpu_recorder(&mut self) {
        if let Some(mut recorder) = self.stacked_gpu_recorder.take() {
            recorder.finish();
            log::info!("StitchCore: GPU stacked-replay recorder detached");
        }
    }

    /// Flush the GPU recorder's buffered bytes to disk. No-op if no
    /// recorder is attached.
    pub fn flush_stacked_gpu_recorder(&mut self) {
        if let Some(ref mut recorder) = self.stacked_gpu_recorder {
            recorder.flush();
        }
    }

    /// Atlas dimensions `(width, height)` the current packer produces,
    /// or `None` when GPU stacked replay is not enabled. Consumers
    /// use this to open an encoder sized for the atlas.
    pub fn stacked_atlas_dims(&self) -> Option<(u32, u32)> {
        self.stacked_packer.as_ref().map(|p| p.atlas_dims())
    }

    /// Pack the GPU stacked-replay atlas from external texture
    /// views (the zero-copy entry point).
    ///
    /// Used by session-layer zero-copy submit paths where source
    /// frames live in shared / imported textures rather than the
    /// renderer's internal plane textures. Call after the stitch
    /// submit has landed; this method encodes a separate command
    /// buffer for the pack + staging copy, submits it, and polls
    /// the triple-buffer ring for a ready atlas to feed to the
    /// attached recorder.
    ///
    /// The storytelling flow (per the project principle - no silent
    /// decisions): the caller chose this path because the source is
    /// GPU-resident. The packer's configured `SourceFormat` was
    /// logged at `enable_gpu_stacked_replay` time. From here on,
    /// every call is just bytes moving through the pipeline, so no
    /// per-frame logging.
    ///
    /// No-op when the packer isn't enabled.
    ///
    /// Hard-coded to the two-camera stereo layout today; extend
    /// when `CameraInput::camera_count() > 2` lands.
    pub fn pack_gpu_stacked_replay_from_views(
        &mut self,
        left: StackedPackSource<'_>,
        right: StackedPackSource<'_>,
    ) {
        crate::profile_scope!("replay_pack_from_views");
        let Some(ref mut packer) = self.stacked_packer else {
            return;
        };
        let gpu = self.pipeline.gpu();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stitch_core_gpu_stacked_pack_ext"),
            });
        let capacity = packer.layout().capacity();
        if capacity >= 1 {
            packer.pack_tile_from_views(gpu, &mut encoder, 0, left);
        }
        if capacity >= 2 {
            packer.pack_tile_from_views(gpu, &mut encoder, 1, right);
        }
        {
            crate::profile_scope!("replay_copy_to_staging");
            packer.copy_to_staging(&mut encoder);
            gpu.queue.submit(Some(encoder.finish()));
        }

        {
            crate::profile_scope!("replay_poll_and_record");
            if let Some(atlas) = packer.poll_ready(gpu)
                && let Some(ref mut recorder) = self.stacked_gpu_recorder
            {
                recorder.record_atlas(&atlas);
            }
        }
    }

    /// Runs the GPU pack from the pipeline's internal plane
    /// textures. Used by CPU-upload submit paths where
    /// `queue.write_texture` has just populated the renderer's
    /// own textures - fires from `submit_frame_yuv*` and from the
    /// session's `process_frame` non-zero-copy branch. Zero-copy
    /// paths take [`Self::pack_gpu_stacked_replay_from_views`]
    /// instead because their source data lives in shared
    /// textures that bypass the renderer's internal planes.
    ///
    /// Delegates through the same pack + poll + record path so
    /// every entry point shares behavior.
    ///
    /// No-op when the packer isn't enabled.
    pub(crate) fn drive_gpu_stacked_pack(&mut self) {
        crate::profile_scope!("replay_drive_pack");
        if self.stacked_packer.is_none() {
            return;
        }
        // Pipeline's plane-view accessors return
        // (y_view, u_or_uv_view, v_or_dummy_view). Build the
        // StackedPackSource variant matching the packer's
        // configured source format - the packer will route to the
        // right shader kernel internally.
        let (ly, lu, lv) = self.pipeline.left_plane_views();
        let (ry, ru, rv) = self.pipeline.right_plane_views();
        // Keep bindings alive across the pack call via locals.
        let (left, right) = match self.pipeline.input_format() {
            InputFormat::Yuv420p => (
                StackedPackSource::Yuv420p {
                    y: &ly,
                    u: &lu,
                    v: &lv,
                },
                StackedPackSource::Yuv420p {
                    y: &ry,
                    u: &ru,
                    v: &rv,
                },
            ),
            InputFormat::Nv12 => (
                StackedPackSource::Nv12 { y: &ly, uv: &lu },
                StackedPackSource::Nv12 { y: &ry, uv: &ru },
            ),
            InputFormat::Bgra => {
                // Shouldn't happen: enable_gpu_stacked_replay
                // rejects BGRA up front. Defensive no-op so the
                // live render loop can't panic on an invariant
                // violation.
                log::error!(
                    "drive_gpu_stacked_pack: packer enabled but pipeline input_format is \
                     BGRA; skipping pack (this is a logic bug in StitchCore)"
                );
                return;
            }
        };
        self.pack_gpu_stacked_replay_from_views(left, right);
    }
}
