# Cylindrical stereo panorama

`Projection::Panorama` gathers two camera inputs directly into a fixed cylindrical output. It reuses the fitted `LShape` rig, KB4 lenses, framing and seam coordinate. It does not refit the cameras or render an intermediate perspective image. The existing `Cylinder` projection has the opposite role: it consumes one already-stitched cylindrical video to render a perspective view.

Use an existing L-shape calibration with the CLI:

```sh
reco stitch left.mp4 right.mp4 -c match.json -o panorama.mp4 \
  --panorama-arc 200 --panorama-vertical-fov 70 --width 2400 --height 964
```

Both extent flags are required together. Alternatively, save the projection in the calibration document and use ordinary `reco stitch` without those flags:

```json
"projection": {
  "type": "panorama",
  "rig": {
    "intersect": 0.55,
    "x_ty": 0.0,
    "x_rz": 0.0,
    "z_rx": 0.0,
    "x_rx": 0.0,
    "z_rz": 0.0,
    "blend_width": 0.05
  },
  "arc_deg": 200.0,
  "vertical_fov_deg": 70.0
}
```

Copy the actual fitted rig values; the numbers above only illustrate the schema. Keep the document's lenses, framing and synchronization. The CLI's existing `--blend` override also applies to panoramas. `CpuExecutor`, `GpuExecutor` and `StitchJob` consume the same document.

## Output geometry

For normalized output coordinates `x,y` in `[-1,1]`, positive y up:

```text
angle = x * arc_radians / 2
ray = (sin(angle), y * tan(vertical_fov_radians / 2), -cos(angle))
```

The inverse of the calibrated zero-pose view matrix transforms this ray into the existing rig frame. Each positive ray-plane intersection is sampled through the same extended-plane support and forward KB4 map used by L-shape rendering. There is no perspective near/far clipping in this output chart. Unsupported pixels remain black. The right source blends over the left in supported overlap; where only the right source exists it stays opaque.

The horizontal arc is `[1,359]` degrees, and vertical FOV is `[1,179]` degrees. `Panorama::validate` enforces these ranges as part of calibration validation. The chart has no wrap seam or polar singularity. Framing tilt/roll set its fixed orientation; per-frame pose/FOV and output aspect do not change its extent. Resolution only controls sampling density. For equal horizontal and vertical pixel scale at the chart centre, choose `width/height = arc_radians / (2*tan(vertical_fov_radians/2))`.

This is a full angular panorama rather than a moving perspective crop, but it is not an automatic fit to every source pixel. Choose the extent explicitly. Large extents preserve more boundary coverage and expose curved black borders; smaller extents crop source coverage. Perspective no-black coverage constraints do not describe this chart: the core starts unconstrained for panoramas and coverage preparation rejects it. The CLI rejects autocam and trajectory options for this fixed output. To pan inside the result, use a separate mono `Cylinder` calibration with `arc_deg` equal to the panorama arc and `radius/surface_height = 1/(2*tan(vertical_fov/2))`.

## Inversion and limits

A camera's source-to-output mapping can be inverted only where the lens's central radial branch is monotonic, the eye is not on the source plane, the ray-plane intersection is positive, and both the finite plane and sensor support the sample. The output chart must also contain that ray. `panorama_supported_camera_domains_round_trip` checks this restricted round trip with lens corrections and nonzero rig/framing rotations; it does not assert that every sensor point is representable.

Overlap has two possible source observations for one output direction. Blending and resampling discard information. Different camera centres, occlusions, time offsets and moving objects prevent a global bijection. No lossless conversion, optical-flow seam correction, exposure matching, depth reconstruction or parallax removal is provided. A bad or unstable calibration remains bad in a wider chart.

`panorama_fixed_extent_support_and_overlap` exercises empty, single-camera and overlap regions, seam behavior and fixed extent. `panorama_gpu_matches_cpu_support_seams_and_optics` compares rendered NV12 pixels across arcs, vertical extents, blend widths, lens correction and range modes. The shared WGSL lens function is exercised against Rust in `wgsl_kb4_matches_rust_kb4_on_theta_grid`; existing L-shape agreement tests cover its other consumers. These tests establish numerical agreement, not performance or optical accuracy on a particular recording.
