/**
 * Generates vertex and fragment shaders for fisheye distortion correction.
 *
 * Creates WebGL shaders that apply fisheye lens distortion correction using a polynomial model.
 * The fragment shader performs radial distortion correction based on camera intrinsic parameters
 * and distortion coefficients.
 *
 * @param {boolean} isLeft  Determines whether to generate shaders for the left or right view.
 *                          - When true: processes the bottom half of the texture (UV.y: 0.5-1.0)
 *                          - When false: processes the top half of the texture (UV.y: 0.0-0.5)
 *
 * @returns {{vertexShader: string, fragmentShader: string}} An object containing:
 *   - vertexShader: GLSL vertex shader code that transforms vertices and passes UV coordinates
 *   - fragmentShader: GLSL fragment shader code that applies fisheye distortion correction
 *
 * @description
 * The fragment shader expects the following uniforms to be set:
 * - uVideo (sampler2D): The input video texture
 * - fx, fy (float): Focal lengths in x and y directions
 * - cx, cy (float): Principal point coordinates (optical center)
 * - d (vec4): Distortion coefficients (k1, k2, k3, k4) for polynomial distortion model
 *
 * The distortion model uses: ```theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)```
 *
 */
const fisheyeShader = (isLeft) => {
  return {
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv * 2.0 - 0.5;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      uniform sampler2D uVideo;
      uniform float fx, fy, cx, cy;
      uniform vec4 d;
      varying vec2 vUv;

      void main() {
        float x = (vUv.x - cx) / fx;
        float y = (vUv.y - cy) / fy;
        float r = sqrt(x*x + y*y);
        float theta = atan(r);
        float theta_d = theta * (1.0 + d.x*pow(theta,2.0) + d.y*pow(theta,4.0) + d.z*pow(theta,6.0) + d.w*pow(theta,8.0));
        float scale = r > 0.0 ? theta_d / r : 1.0;
        x *= scale; y *= scale;

        vec2 distortedUV;
        distortedUV.x = fx * x + cx;
        distortedUV.y = fy * y + cy;

        if (${isLeft ? "false" : "true"}) {
          distortedUV.y *= 0.5;
        } else {
          distortedUV.y = distortedUV.y * 0.5 + 0.5;
        }

        if (distortedUV.x < 0.0 || distortedUV.x > 1.0 || distortedUV.y < ${isLeft ? "0.5" : "0.0"} || distortedUV.y > ${isLeft ? "1.0" : "0.5"}) {
          gl_FragColor = vec4(0.0);
        } else {
          gl_FragColor = texture2D(uVideo, distortedUV);
        }
      }
    `,
  };
};

export default fisheyeShader;
