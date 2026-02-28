/**
 * Barrel distortion correction shader for panoramic cameras.
 *
 * Supports two modes:
 * 1. Full-image mode (splitHalf = 0): corrects the entire panorama as one unit
 * 2. Split-half mode (splitHalf = 1): per-lens correction for dual-lens cameras
 *
 * Uses Brown-Conrady polynomial: r_src = r_out * (1 + k1*r^2 + k2*r^4 + k3*r^6 + k4*r^8)
 * With d=[0,0,0,0] the shader is pure passthrough.
 * Positive k1 corrects barrel distortion (bowed-out lines).
 */
const dewarpShader = () => {
	return {
		vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
		fragmentShader: `
      precision highp float;
      uniform sampler2D uVideo;

      uniform float splitPoint;
      uniform float blendWidth;
      uniform float splitHalf; // 0.0 = full-image mode, 1.0 = split-half mode

      // Left/full-image lens intrinsics
      uniform float lFx, lFy, lCx, lCy;
      uniform vec4 lD;

      // Right lens intrinsics (only used in split-half mode)
      uniform float rFx, rFy, rCx, rCy;
      uniform vec4 rD;

      varying vec2 vUv;

      vec2 undistort(vec2 inputUV, float fx, float fy, float cx, float cy, vec4 d) {
        float x = (inputUV.x - cx) / fx;
        float y = (inputUV.y - cy) / fy;
        float r2 = x * x + y * y;
        float scale = 1.0 + d.x * r2 + d.y * r2 * r2 + d.z * r2 * r2 * r2 + d.w * r2 * r2 * r2 * r2;
        return vec2(fx * x * scale + cx, fy * y * scale + cy);
      }

      void main() {
        vec2 distortedUV;

        if (splitHalf < 0.5) {
          // Full-image mode: correct entire panorama as one unit
          distortedUV = undistort(vUv, lFx, lFy, lCx, lCy, lD);
        } else {
          // Split-half mode: per-lens correction
          bool isLeft = vUv.x < splitPoint;

          if (isLeft) {
            vec2 localUV = vec2(vUv.x / splitPoint, vUv.y);
            vec2 localDistorted = undistort(localUV, lFx, lFy, lCx, lCy, lD);
            distortedUV = vec2(localDistorted.x * splitPoint, localDistorted.y);
          } else {
            vec2 localUV = vec2((vUv.x - splitPoint) / (1.0 - splitPoint), vUv.y);
            vec2 localDistorted = undistort(localUV, rFx, rFy, rCx, rCy, rD);
            distortedUV = vec2(localDistorted.x * (1.0 - splitPoint) + splitPoint, localDistorted.y);
          }

          // Blend zone at stitch seam
          if (blendWidth > 0.0) {
            float distFromSplit = abs(vUv.x - splitPoint);
            if (distFromSplit < blendWidth) {
              vec2 otherDistortedUV;
              if (isLeft) {
                vec2 localUV = vec2((vUv.x - splitPoint) / (1.0 - splitPoint), vUv.y);
                vec2 localDistorted = undistort(localUV, rFx, rFy, rCx, rCy, rD);
                otherDistortedUV = vec2(localDistorted.x * (1.0 - splitPoint) + splitPoint, localDistorted.y);
              } else {
                vec2 localUV = vec2(vUv.x / splitPoint, vUv.y);
                vec2 localDistorted = undistort(localUV, lFx, lFy, lCx, lCy, lD);
                otherDistortedUV = vec2(localDistorted.x * splitPoint, localDistorted.y);
              }

              float t = smoothstep(0.0, blendWidth, distFromSplit);
              if (otherDistortedUV.x >= 0.0 && otherDistortedUV.x <= 1.0 &&
                  otherDistortedUV.y >= 0.0 && otherDistortedUV.y <= 1.0) {
                vec4 primary = texture2D(uVideo, distortedUV);
                vec4 secondary = texture2D(uVideo, otherDistortedUV);
                gl_FragColor = mix(secondary, primary, t);
                return;
              }
            }
          }
        }

        // Bounds check
        if (distortedUV.x < 0.0 || distortedUV.x > 1.0 || distortedUV.y < 0.0 || distortedUV.y > 1.0) {
          gl_FragColor = vec4(0.0);
          return;
        }

        gl_FragColor = texture2D(uVideo, distortedUV);
      }
    `,
	};
};

export default dewarpShader;
