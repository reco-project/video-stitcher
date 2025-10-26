/* eslint-disable react/prop-types */ // TODO: remove this line and define prop types
//import PropTypes from "prop-types";
import * as THREE from "three";
import React, { useEffect, useRef } from "react";
import { useViewerStore } from "../stores/store.js";
import { Canvas } from "@react-three/fiber";
import { useVideoTexture } from "@react-three/drei";
import fisheyeShader from "../shaders/fisheye.js";
import { ErrorBoundary } from "react-error-boundary";
import Controls from "./Controls.jsx";
import { formatUniforms } from "../utils/utils.js";

const VideoPlane = ({ texture, isLeft }) => {
  const selectedMatch = useViewerStore((s) => s.selectedMatch);
  const params = selectedMatch ? selectedMatch.params : {};
  const u = selectedMatch ? selectedMatch.uniforms : {};
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
      <shaderMaterial
        uniforms={formatUniforms(u, texture)}
        {...fisheyeShader(isLeft)}
      />
    </mesh>
  );
};

const VideoPanorama = () => {
  const selectedMatch = useViewerStore((s) => s.selectedMatch);
  const src = selectedMatch ? selectedMatch.src : null;
  const setVideoRef = useViewerStore((s) => s.setVideoRef);
  const clearVideoRef = useViewerStore((s) => s.clearVideoRef);

  const texture = useVideoTexture(src || "", {
    muted: true,
    loop: true,
    playsInline: true,
    start: !!src,
    unsuspend: "canplay",
  });

  // Register the underlying HTMLVideoElement (texture.image) in the store so controls can use it.
  React.useEffect(() => {
    const v = texture?.image;
    if (v) {
      setVideoRef(v);
    }
    return () => {
      clearVideoRef();
    };
  }, [texture, setVideoRef, clearVideoRef]);

  if (!src) return null;

  return (
    <group>
      <VideoPlane texture={texture} isLeft={true} />
      <VideoPlane texture={texture} isLeft={false} />
    </group>
  );
};

const Viewer = ({ selectedMatch }) => {
  const containerRef = useRef(null);
  const setSelectedMatch = useViewerStore((s) => s.setSelectedMatch);

  useEffect(() => {
    setSelectedMatch(selectedMatch);
  }, [selectedMatch, setSelectedMatch]); // TODO: missing validation

  const cameraAxisOffset = selectedMatch.params.cameraAxisOffset; // TODO: validate. No use of "?." here to avoid silent errors.

  // TODO: this should be moved to the video player controls
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
