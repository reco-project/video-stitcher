/**
 * Lens profiles for panoramic cameras.
 *
 * Structure:
 * - width/height: full stitched image resolution
 * - splitHalf: false = full-image correction, true = per-half correction
 * - splitPoint: X position of stitch seam (0-1), used in split-half mode
 * - blendWidth: blend zone width at seam (0-1, 0 = hard edge)
 * - left: primary lens calibration (used for full-image mode too)
 *   - fx, fy: coordinate scale in pixels (smaller = stronger correction at edges)
 *   - cx, cy: principal point in pixels
 *   - d: [k1, k2, k3, k4] Brown-Conrady distortion coefficients
 *     Positive k1 corrects barrel distortion. d=[0,0,0,0] = passthrough.
 * - right: second lens calibration (only used in split-half mode)
 */

const profiles = {
	none: {
		label: 'No Dewarping',
		width: 4096,
		height: 1800,
		splitHalf: false,
		splitPoint: 0.5,
		blendWidth: 0,
		left: {
			fx: 4096,
			fy: 1800,
			cx: 2048,
			cy: 900,
			d: [0, 0, 0, 0],
		},
		right: {
			fx: 4096,
			fy: 1800,
			cx: 2048,
			cy: 900,
			d: [0, 0, 0, 0],
		},
	},
	dahua_b180: {
		label: 'Dahua IPC-Color4K-B180',
		width: 4096,
		height: 1800,
		splitHalf: false,
		splitPoint: 0.5,
		blendWidth: 0,
		// Full-image barrel correction.
		// fx=4096 (full width) means x ranges [-0.5, 0.5], fy=1800 same.
		// At edge: r2 ≈ 0.25, with k1=0.5: scale = 1.125 (12.5% correction)
		// At corner: r2 ≈ 0.5, with k1=0.5: scale = 1.25 (25% correction)
		left: {
			fx: 4096,
			fy: 1800,
			cx: 2048,
			cy: 900,
			d: [0, 0, 0, 0],
		},
		right: {
			fx: 4096,
			fy: 1800,
			cx: 2048,
			cy: 900,
			d: [0, 0, 0, 0],
		},
	},
	reolink_duo3: {
		label: 'Reolink Duo 3 PoE',
		width: 1280,
		height: 360,
		splitHalf: false,
		splitPoint: 0.5,
		blendWidth: 0,
		// Cylindrical correction handled by half-cylinder geometry.
		// d=[0,0,0,0] = no residual barrel correction.
		left: {
			fx: 1280,
			fy: 360,
			cx: 640,
			cy: 180,
			d: [0, 0, 0, 0],
		},
		right: {
			fx: 1280,
			fy: 360,
			cx: 640,
			cy: 180,
			d: [0, 0, 0, 0],
		},
	},
};

export default profiles;
