/**
 * Generates vertex and fragment shaders for fisheye distortion correction with color correction.
 *
 * Creates WebGL shaders that apply fisheye lens distortion correction using a polynomial model,
 * plus color correction for matching exposure/white balance between cameras.
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
 * - brightness (float): Brightness adjustment (-1 to 1, default 0)
 * - contrast (float): Contrast multiplier (0 to 2, default 1)
 * - saturation (float): Saturation multiplier (0 to 2, default 1)
 * - colorBalance (vec3): RGB gain multipliers (default 1,1,1)
 * - temperature (float): Color temperature shift (-1 to 1, default 0)
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
      
      // Color correction uniforms
      uniform float brightness;
      uniform float contrast;
      uniform float saturation;
      uniform vec3 colorBalance;
      uniform float temperature;
      
      varying vec2 vUv;

      // Convert RGB to HSL
      vec3 rgb2hsl(vec3 color) {
        float maxC = max(max(color.r, color.g), color.b);
        float minC = min(min(color.r, color.g), color.b);
        float l = (maxC + minC) / 2.0;
        
        if (maxC == minC) {
          return vec3(0.0, 0.0, l);
        }
        
        float d = maxC - minC;
        float s = l > 0.5 ? d / (2.0 - maxC - minC) : d / (maxC + minC);
        
        float h;
        if (maxC == color.r) {
          h = (color.g - color.b) / d + (color.g < color.b ? 6.0 : 0.0);
        } else if (maxC == color.g) {
          h = (color.b - color.r) / d + 2.0;
        } else {
          h = (color.r - color.g) / d + 4.0;
        }
        h /= 6.0;
        
        return vec3(h, s, l);
      }

      // Convert HSL to RGB
      float hue2rgb(float p, float q, float t) {
        if (t < 0.0) t += 1.0;
        if (t > 1.0) t -= 1.0;
        if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
        if (t < 1.0/2.0) return q;
        if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
        return p;
      }

      vec3 hsl2rgb(vec3 hsl) {
        if (hsl.y == 0.0) {
          return vec3(hsl.z);
        }
        
        float q = hsl.z < 0.5 ? hsl.z * (1.0 + hsl.y) : hsl.z + hsl.y - hsl.z * hsl.y;
        float p = 2.0 * hsl.z - q;
        
        return vec3(
          hue2rgb(p, q, hsl.x + 1.0/3.0),
          hue2rgb(p, q, hsl.x),
          hue2rgb(p, q, hsl.x - 1.0/3.0)
        );
      }

      // Apply color temperature (shift between warm/cool)
      vec3 applyTemperature(vec3 color, float temp) {
        // Warm = more red/yellow, Cool = more blue
        color.r += temp * 0.1;
        color.b -= temp * 0.1;
        return clamp(color, 0.0, 1.0);
      }

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

        if (${isLeft ? 'false' : 'true'}) {
          distortedUV.y *= 0.5;
        } else {
          distortedUV.y = distortedUV.y * 0.5 + 0.5;
        }

        if (distortedUV.x < 0.0 || distortedUV.x > 1.0 || distortedUV.y < ${isLeft ? '0.5' : '0.0'} || distortedUV.y > ${isLeft ? '1.0' : '0.5'}) {
          gl_FragColor = vec4(0.0);
        } else {
          vec4 texColor = texture2D(uVideo, distortedUV);
          vec3 color = texColor.rgb;
          
          // Apply color balance (RGB gains)
          color *= colorBalance;
          
          // Apply temperature
          color = applyTemperature(color, temperature);
          
          // Apply brightness
          color += brightness;
          
          // Apply contrast (around mid-gray)
          color = (color - 0.5) * contrast + 0.5;
          
          // Apply saturation
          vec3 hsl = rgb2hsl(color);
          hsl.y *= saturation;
          color = hsl2rgb(hsl);
          
          // Clamp final result
          color = clamp(color, 0.0, 1.0);
          
          gl_FragColor = vec4(color, texColor.a);
        }
      }
    `,
  };
};

export default fisheyeShader;
