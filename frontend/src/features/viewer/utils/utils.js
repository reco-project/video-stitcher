import * as THREE from 'three';

// Default color correction values
export const DEFAULT_COLOR_CORRECTION = {
	brightness: 0,
	contrast: 1,
	saturation: 1,
	colorBalance: [1, 1, 1], // RGB gains
	temperature: 0,
};

// Consideration: This function may be useless if we already store uniforms in the expected format.
// But I like to think that this would benefit modularity and reusability.

/**
 * This function converts the uniforms object to the format required by the shader material.
 * @param {*} u The uniforms object containing width, height, fx, fy, cx, cy, d
 * @param {*} texture The video texture to be used
 * @param {*} colorCorrection Optional color correction parameters
 * @returns An object with the converted uniforms
 */
export function formatUniforms(u, texture, colorCorrection = {}) {
	if (!u || !u.width || !u.height || !u.fx || !u.fy || !u.cx || !u.cy) {
		console.error('Invalid uniforms:', u);
		throw new Error('Missing required uniform parameters');
	}

	if (!u.d || !Array.isArray(u.d) || u.d.length !== 4) {
		console.error('Invalid distortion coefficients:', u.d);
		throw new Error('Distortion coefficients must be an array of 4 numbers');
	}

	// Merge with defaults
	const cc = { ...DEFAULT_COLOR_CORRECTION, ...colorCorrection };

	const width = u.width;
	const height = u.height;
	return {
		uVideo: { value: texture },
		fx: { value: u.fx / width },
		fy: { value: u.fy / height },
		cx: { value: u.cx / width },
		cy: { value: u.cy / height },
		d: { value: new THREE.Vector4(...u.d) },
		// Color correction uniforms
		brightness: { value: cc.brightness },
		contrast: { value: cc.contrast },
		saturation: { value: cc.saturation },
		colorBalance: { value: new THREE.Vector3(...cc.colorBalance) },
		temperature: { value: cc.temperature },
	};
}
