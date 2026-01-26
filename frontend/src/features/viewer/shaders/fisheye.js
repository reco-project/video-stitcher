/**
 * Generates vertex and fragment shaders for fisheye distortion correction with color correction.
 *
 * Creates WebGL shaders that apply fisheye lens distortion correction using a polynomial model,
 * plus color correction using Reinhard color transfer in LAB color space for better matching
 * between cameras.
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
 * - labScale (vec3): LAB color space scale factors for Reinhard transfer
 * - labOffset (vec3): LAB color space offset values for Reinhard transfer
 *
 * Legacy uniforms (still supported for backward compatibility):
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
      
      // LAB-based Reinhard color transfer uniforms
      uniform vec3 labScale;
      uniform vec3 labOffset;
      
      // Legacy color correction uniforms (for backward compatibility)
      uniform float brightness;
      uniform float contrast;
      uniform float saturation;
      uniform vec3 colorBalance;
      uniform float temperature;
      
      varying vec2 vUv;

      // RGB to LAB conversion (OpenCV-compatible range: L: 0-255, a: 0-255, b: 0-255)
      // This matches OpenCV's COLOR_BGR2LAB output range for consistency with backend
      vec3 rgb2xyz(vec3 rgb) {
        // sRGB to linear RGB
        vec3 lin = mix(
          rgb / 12.92,
          pow((rgb + 0.055) / 1.055, vec3(2.4)),
          step(0.04045, rgb)
        );
        // Linear RGB to XYZ (D65 illuminant)
        // GLSL mat3 is column-major: mat3(col0, col1, col2)
        // For m * v: result = v.x*col0 + v.y*col1 + v.z*col2
        // We want: X = 0.4124564*R + 0.3575761*G + 0.1804375*B, etc.
        // So columns are: [X coeffs], [Y coeffs], [Z coeffs] for R, G, B
        mat3 m = mat3(
          0.4124564, 0.2126729, 0.0193339,  // R coefficients (column 0)
          0.3575761, 0.7151522, 0.1191920,  // G coefficients (column 1)
          0.1804375, 0.0721750, 0.9503041   // B coefficients (column 2)
        );
        return m * lin * 100.0;
      }

      vec3 xyz2lab(vec3 xyz) {
        // D65 reference white
        vec3 ref = vec3(95.047, 100.0, 108.883);
        vec3 f = xyz / ref;
        vec3 ft = mix(
          (903.3 * f + 16.0) / 116.0,
          pow(f, vec3(1.0/3.0)),
          step(0.008856, f)
        );
        // Standard LAB: L: 0-100, a/b: roughly -128 to 127
        float L = 116.0 * ft.y - 16.0;
        float a = 500.0 * (ft.x - ft.y);
        float b = 200.0 * (ft.y - ft.z);
        
        // Convert to OpenCV LAB range: L: 0-255, a: 0-255, b: 0-255
        return vec3(
          L * (255.0 / 100.0),
          a + 128.0,
          b + 128.0
        );
      }

      vec3 lab2xyz(vec3 labOcv) {
        // Convert from OpenCV LAB range to standard LAB
        float L = labOcv.x * (100.0 / 255.0);
        float a = labOcv.y - 128.0;
        float b = labOcv.z - 128.0;
        
        float fy = (L + 16.0) / 116.0;
        float fx = a / 500.0 + fy;
        float fz = fy - b / 200.0;
        
        vec3 f = vec3(fx, fy, fz);
        vec3 f3 = f * f * f;
        vec3 xyz = mix(
          (116.0 * f - 16.0) / 903.3,
          f3,
          step(0.008856, f3)
        );
        // D65 reference white
        return xyz * vec3(95.047, 100.0, 108.883);
      }

      vec3 xyz2rgb(vec3 xyz) {
        // XYZ to linear RGB
        // GLSL mat3 is column-major: mat3(col0, col1, col2)
        // For m * v: result = v.x*col0 + v.y*col1 + v.z*col2
        // We want: R = 3.2404542*X - 1.5371385*Y - 0.4985314*Z, etc.
        mat3 m = mat3(
           3.2404542, -0.9692660,  0.0556434,  // X coefficients (column 0)
          -1.5371385,  1.8760108, -0.2040259,  // Y coefficients (column 1)
          -0.4985314,  0.0415560,  1.0572252   // Z coefficients (column 2)
        );
        vec3 lin = m * (xyz / 100.0);
        // Linear RGB to sRGB
        return mix(
          lin * 12.92,
          1.055 * pow(lin, vec3(1.0/2.4)) - 0.055,
          step(0.0031308, lin)
        );
      }

      vec3 rgb2lab(vec3 rgb) {
        return xyz2lab(rgb2xyz(rgb));
      }

      vec3 lab2rgb(vec3 lab) {
        return xyz2rgb(lab2xyz(lab));
      }

      // Apply Reinhard color transfer in LAB space (OpenCV range: 0-255)
      // Matches: lab_t = (lab - src_mean) / src_std * tgt_std + tgt_mean
      // Which is: lab_t = lab * scale + offset
      vec3 applyReinhardLAB(vec3 rgb, vec3 scale, vec3 offset) {
        // Skip if identity transform
        if (scale == vec3(1.0) && offset == vec3(0.0)) {
          return rgb;
        }
        vec3 lab = rgb2lab(rgb);  // Now in OpenCV range: 0-255
        lab = lab * scale + offset;
        // Clamp LAB values to valid OpenCV range (0-255)
        lab = clamp(lab, 0.0, 255.0);
        return clamp(lab2rgb(lab), 0.0, 1.0);
      }

      // Legacy: Convert RGB to HSL
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

      // Legacy: Convert HSL to RGB
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

      // Legacy: Apply color temperature
      vec3 applyTemperature(vec3 color, float temp) {
        color.r += temp * 0.1;
        color.b -= temp * 0.1;
        return clamp(color, 0.0, 1.0);
      }

      // Legacy color correction (brightness, contrast, saturation, etc.)
      vec3 applyLegacyCorrection(vec3 color) {
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
          
          // Apply LAB-based Reinhard color transfer (primary method)
          color = applyReinhardLAB(color, labScale, labOffset);
          
          // Apply legacy corrections if any are non-default
          bool hasLegacy = brightness != 0.0 || contrast != 1.0 || saturation != 1.0 || 
                          colorBalance != vec3(1.0) || temperature != 0.0;
          if (hasLegacy) {
            color = applyLegacyCorrection(color);
          }
          
          gl_FragColor = vec4(color, texColor.a);
        }
      }
    `,
  };
};

export default fisheyeShader;
