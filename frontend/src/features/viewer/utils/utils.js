import * as THREE from 'three';

// Consideration: This function may be useless if we already store uniforms in the expected format.
// But I like to think that this would benefit modularity and reusability.

/**
 * This function converts the uniforms object to the format required by the shader material.
 * @param {*} u The uniforms object containing width, height, fx, fy, cx, cy, d
 * @param {*} texture The video texture to be used
 * @returns An object with the converted uniforms
 */
export function formatUniforms(u, texture) {
	const width = u.width;
	const height = u.height;
	return {
		uVideo: { value: texture },
		fx: { value: u.fx / width },
		fy: { value: u.fy / height },
		cx: { value: u.cx / width },
		cy: { value: u.cy / height },
		d: { value: new THREE.Vector4(...u.d) },
	};
}
