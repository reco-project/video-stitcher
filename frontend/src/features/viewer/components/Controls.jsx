import { useEffect, useRef } from "react";
import { useThree } from "@react-three/fiber";
import * as THREE from "three";

const Controls = () => {
  const { camera, gl } = useThree();
  const canvas = gl.domElement;
  const dragging = useRef(false);
  const lastX = useRef(0);
  const lastY = useRef(0);

  const isInvertedPitch = true;
  const panSensitivity = 0.005; // adjust pan speed
  const zoomSensitivity = 0.05; // adjust zoom speed
  const minFov = 30;
  const maxFov = 75;

  /* Quaternions and vectors reused for performance, right? */
  const yawQuat = new THREE.Quaternion();
  const pitchQuat = new THREE.Quaternion();
  const rightVec = new THREE.Vector3(1, 0, 0);

  // TODO: adjust these ranges dynamically based on FOV and distortion
  const pitchRange = THREE.MathUtils.degToRad(20); // vertical
  const yawRange = THREE.MathUtils.degToRad(140); // horizontal

  const minPitch = -pitchRange / 2 - THREE.MathUtils.degToRad(10);
  const maxPitch = pitchRange / 2 - THREE.MathUtils.degToRad(10);
  const minYaw = -yawRange / 2 + THREE.MathUtils.degToRad(45);
  const maxYaw = yawRange / 2 + THREE.MathUtils.degToRad(45);

  useEffect(() => {

    const onPointerDown = (e) => {
      dragging.current = true;
      lastX.current = e.clientX;
      lastY.current = e.clientY;
    };

    const onPointerUp = () => {
      dragging.current = false;
    };

    const onPointerMove = (e) => {
      if (!dragging.current) return;

      const deltaX = e.clientX - lastX.current;
      const deltaY = e.clientY - lastY.current;
      lastX.current = e.clientX;
      lastY.current = e.clientY;

      // yaw: rotate around world Y axis
      yawQuat.setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        deltaX * panSensitivity
      );
      camera.quaternion.premultiply(yawQuat);

      // pitch: rotate around camera's local right axis
      rightVec.set(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
      pitchQuat.setFromAxisAngle(
        rightVec,
        (isInvertedPitch ? 1 : -1) * deltaY * panSensitivity
      );
      camera.quaternion.premultiply(pitchQuat);

      // clamp by converting to Euler, clamping, and converting back
      const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
      euler.x = THREE.MathUtils.clamp(euler.x, minPitch, maxPitch);
      euler.y = THREE.MathUtils.clamp(euler.y, minYaw, maxYaw);
      camera.quaternion.setFromEuler(euler);
    };

    const onWheel = (e) => {
      e.preventDefault();
      camera.fov = THREE.MathUtils.clamp(
        camera.fov + e.deltaY * zoomSensitivity,
        minFov,
        maxFov
      );
      camera.updateProjectionMatrix();
    };

    // attached to window to capture events outside the canvas
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointermove", onPointerMove);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointermove", onPointerMove);
    };
  }, [camera]);
};

export default Controls;
