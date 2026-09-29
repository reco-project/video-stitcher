# Core API friction

## Mixed source colour ranges

The renderer has one `set_full_range` flag for all camera inputs. File sources therefore reject a mixture of full-range and limited-range cameras instead of rendering one camera incorrectly. Per-input range handling belongs in the core renderer, not in consumer-side pixel conversions. The CPU-source rejection is covered by `ffmpeg::decoder::tests::cpu_source_rejects_mixed_camera_ranges_in_both_orders`.

## Non-default CUDA decode devices

File decoders request CUDA device 0. Core validates that native imports match both this device and the current allocation device; other render devices use host-backed transfer. Native processing on another CUDA device needs explicit device selection shared by decoding, allocation and worker contexts, not a consumer-side thread-local device switch.

## Per-camera identity and retention for live pairing

[`SourceFrame`](../reco-core/src/source.rs) carries per-camera capture times independently of output presentation time. GStreamer host/NVMM capture preserves each camera's running-time timestamp; [`run_detection_frames`](../reco-core/src/core/pose.rs) remembers each camera's last successfully analysed image. Reused images are not new tracking evidence, but an image first displayed during a skipped detection tick can still be analysed. This is covered by `reused_camera_images_are_inferred_once_in_immediate_and_lookahead_sessions` and `capture_identity_handles_retry_empty_results_and_invalidation` in the core session tests. Sources without capture times still treat every inference submission as a new image.

`CaptureTiming` declares an independent source-clock assessment instant and an optional image-age bound. Session tracking expires held observations using that bound plus one scheduled detection interval, including skipped inference ticks; identity survives expiry. `observation_expiry_uses_source_age_before_immediate_and_lookahead_tracking` and `expiry_preserves_identity_and_cannot_revive_an_old_measurement` cover this contract. Sources declaring no age bound retain identity-only behavior. Tracker prediction time is still processing elapsed time, separate from this observation-age gate.

CPU `FrameSet` variants retain each image through `Arc<Yuv420pData>`/`Arc<Nv12Data>`; mono libcamera shares one image for its duplicated view. The `shared_host_images_survive_source_release_and_match_owned_delivery` regression covers both host formats through immediate/lookahead processing and threaded output. Optional GStreamer matching shares these handles; `bounded_live_reuse_survives_inference_lookahead_and_encoding` checks both host formats through encoded output. NVMM also has retained frame ownership, but live reuse still needs a verified native-buffer retention budget before enabling it in camera consumers. Retention alone does not make repeated observations fresh.
