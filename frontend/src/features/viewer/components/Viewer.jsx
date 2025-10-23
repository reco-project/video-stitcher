//import PropTypes from "prop-types";
import * as THREE from "three";
import React, { useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { useVideoTexture } from "@react-three/drei";
import fisheyeShader from "./shaders/fisheye";
import { ErrorBoundary } from "react-error-boundary";
import Controls from "./Controls.jsx";

/*Viewer.propTypes = {
  match: PropTypes.shape({
    src: PropTypes.string.isRequired,
  }).isRequired,
  settings: PropTypes.shape({
    cam_d: PropTypes.number.isRequired,
  }).isRequired,
};*/

const VideoPlane = ({ texture, isLeft }) => {
  const params = {
    intersect: 0.5472022558355283,
    zRx: -0.04131782452879521,
    xTy: -0.002024608962576278,
    xRz: -0.01969244886237673,
  };

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
  const planeWidth = 1;
  const aspect = 16 / 9;

  const position = isLeft
    ? [0, 0, (planeWidth / 2) * (1 - params.intersect)]
    : [(planeWidth / 2) * (1 - params.intersect), params.xTy, 0];
  const rotation = isLeft
    ? [params.zRx, THREE.MathUtils.degToRad(90), 0]
    : [0, 0, params.xRz];

  return (
    <mesh position={position} rotation={rotation}>
      <planeGeometry args={[planeWidth, planeWidth / aspect]} />
      <shaderMaterial uniforms={uniforms} {...fisheyeShader(isLeft)} />
    </mesh>
  );
};

const VideoPanorama = () => {
  const texture = useVideoTexture(
    "https://storage.googleapis.com/reco-bucket-processed/stacked_genolier.mp4",
    {
      muted: true,
      loop: true,
      playsInline: true,
      start: true,
      unsuspend: "canplay",
    } // TODO: should remove things already defaulted
  );
  return (
    <group>
      <VideoPlane texture={texture} isLeft={true} />
      <VideoPlane texture={texture} isLeft={false} />
    </group>
  );
};

const Viewer = ({ cameraAxisOffset }) => {
  const containerRef = useRef(null);

  const enterFullscreen = () => {
    const el = containerRef.current;
    if (!el) return;
    if (el.requestFullscreen) {
      el.requestFullscreen();
    } else if (el.webkitRequestFullscreen) {
      el.webkitRequestFullscreen();
    } else if (el.msRequestFullscreen) {
      el.msRequestFullscreen();
    }
  };

  return (
    <ErrorBoundary fallback={<div>Error loading video panorama</div>}>
      <Canvas
        ref={containerRef}
        camera={{
          position: [cameraAxisOffset, 0, cameraAxisOffset],
          fov: 75,
          near: 0.01,
          far: 5,
        }}
      >
        <Controls />
        <VideoPanorama />
      </Canvas>
      <button
        onClick={enterFullscreen}
        className="absolute bottom-2 right-2 px-2 py-1 rounded bg-purple-600 text-white"
      >
        Fullscreen
      </button>
    </ErrorBoundary>
  );
};

export default Viewer;
