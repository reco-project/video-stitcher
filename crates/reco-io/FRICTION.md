# Core API friction

## Mixed source colour ranges

The renderer has one `set_full_range` flag for all camera inputs. File sources therefore reject a mixture of full-range and limited-range cameras instead of rendering one camera incorrectly. Per-input range handling belongs in the core renderer, not in consumer-side pixel conversions. The CPU-source rejection is covered by `ffmpeg::decoder::tests::cpu_source_rejects_mixed_camera_ranges_in_both_orders`.

## Non-default CUDA decode devices

File decoders request CUDA device 0. Core validates that native imports match both this device and the current allocation device; other render devices use host-backed transfer. Native processing on another CUDA device needs explicit device selection shared by decoding, allocation and worker contexts, not a consumer-side thread-local device switch.
