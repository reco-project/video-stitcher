# Panorama Viewer

A browser-based viewer for pre-stitched 180-degree panoramic camera footage with real-time dewarping and pan/zoom controls.

## Use Case

Many panoramic security and sports cameras (e.g., Dahua IPC-Color4K-B180, Reolink Duo 3 PoE) produce a single pre-stitched wide-angle image from dual internal sensors. This image uses a **cylindrical projection**, which causes straight lines in the real world (like soccer field sidelines) to appear curved when displayed on a flat screen.

The panorama viewer solves this by:

1. Mapping the video texture onto a **half-cylinder geometry** inside a WebGL scene
2. Placing a perspective camera at the cylinder's center
3. Letting the user pan and zoom across the full 180-degree field of view

The perspective camera naturally converts the cylindrical projection to a rectilinear one, making straight lines appear straight in the viewport -- no polynomial distortion correction needed.

## How It Works

### Geometry

The viewer creates a half-cylinder (`createHalfCylinderGeometry`) with the panoramic video texture mapped to its inner surface. The camera sits at the origin looking outward through the cylinder wall (rendered with `THREE.BackSide`).

This is geometrically equivalent to "unrolling" the cylindrical projection back onto a curved surface and viewing it from the correct vantage point.

### Residual Correction

For cameras that have additional barrel distortion beyond the cylindrical projection, a Brown-Conrady polynomial shader (`dewarpShader.js`) can apply per-pixel correction. The shader supports:

- **Full-image mode**: Corrects the entire panorama as one unit
- **Split-half mode**: Independent correction for each lens half, with configurable blend at the stitch seam

With `d=[0,0,0,0]`, the shader is pure passthrough (no correction applied).

### Camera Profiles

Lens profiles are defined in `profiles.js` with intrinsics (`fx`, `fy`, `cx`, `cy`) and distortion coefficients (`d`). Users can also override coefficients via the UI for calibration.

## Usage

Navigate to `/panorama` in the app, or use a direct URL with a video source:

```
/panorama?src=/path/to/video.mp4
```

### Controls

- **Click and drag** to pan across the panorama
- **Scroll wheel** to zoom in/out (adjusts FOV from 15 to 90 degrees)
- **Space/K** to play/pause
- **J/L or arrow keys** to seek +/- 5 seconds
- **F** for fullscreen

### Adding Camera Profiles

Add new profiles to `profiles.js` following the existing pattern:

```js
my_camera: {
    label: 'My Camera Model',
    width: 4096,        // Full stitched image width
    height: 1800,       // Full stitched image height
    splitHalf: false,    // true for per-lens correction
    splitPoint: 0.5,     // Stitch seam position (0-1)
    blendWidth: 0,       // Blend zone width at seam
    left: {
        fx: 4096, fy: 1800,  // Coordinate scale in pixels
        cx: 2048, cy: 900,   // Principal point in pixels
        d: [0, 0, 0, 0],     // [k1, k2, k3, k4] distortion coefficients
    },
    right: { /* same structure, only used in split-half mode */ },
},
```

Use the "Override" checkbox in the UI to interactively adjust coefficients until field lines appear straight, then copy the values into a profile.
