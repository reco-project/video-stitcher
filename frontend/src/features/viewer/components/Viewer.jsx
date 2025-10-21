import React from "react";
//import PropTypes from "prop-types";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { useVideoTexture } from "@react-three/drei";
import fisheyeShader from "./shaders/fisheye";
import { ErrorBoundary } from "react-error-boundary";

/*Viewer.propTypes = {
  match: PropTypes.shape({
    src: PropTypes.string.isRequired,
  }).isRequired,
  settings: PropTypes.shape({
    cam_d: PropTypes.number.isRequired,
  }).isRequired,
};*/

const VideoPlane = ({ texture, isLeft }) => {
  function createUniforms() {
    const width = 3840;
    const height = 2160;
    return {
      uVideo: { value: texture },
      fx: { value: 1796.3208206894308 / width },
      fy: { value: 1797.22277342282 / height },
      cx: { value: 1919.372365976781 / width },
      cy: { value: 1063.171593155705 / height },
      d: {
        value: new THREE.Vector4(0.03421388, 0.0676732, -0.0740897, 0.02994442),
      },
    };
  }

  const uniforms = createUniforms();
  return (
    <mesh position={[0, 0, 0]}>
      <planeGeometry width={1} height={9 / 16} />
      <shaderMaterial uniforms={uniforms} {...fisheyeShader(isLeft)} />
    </mesh>
  );
};

const VideoPanorama = ({}) => {
  const texture = useVideoTexture(
    "https://storage.googleapis.com/reco-bucket-processed/stacked_genolier.mp4",
    {
      muted: true,
      loop: true,
      playsInline: true,
      start: true,
      unsuspend: "canplay",
    }
  );
  return (
    <group>
      <VideoPlane texture={texture} isLeft={true} />
      <VideoPlane texture={texture} isLeft={false} />
    </group>
  );
};

const Viewer = ({ cameraAxisOffset }) => {
  return (
    <ErrorBoundary fallback={<div>Error loading video panorama</div>}>
      <Canvas camera={{ position: [cameraAxisOffset, 0, cameraAxisOffset], fov: 75, near: 0.01, far: 5 }}>
        <VideoPanorama />
      </Canvas>
    </ErrorBoundary>
  );
};

export default Viewer;
