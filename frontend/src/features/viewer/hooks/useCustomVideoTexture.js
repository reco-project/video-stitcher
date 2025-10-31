import { useEffect, useState } from "react";
import * as THREE from "three";
import { Hls, isSupported } from "hls.js";
import { useViewerStore } from "../stores/store.js";

/** 
 * 
 * This hook creates and manages a video texture for a given video source URL.
 * It supports both regular video sources and HLS (.m3u8) streams using hls.js.
 * The video element is created and controlled within the hook, and the resulting 
 * video texture is returned for use in 3D scenes.
 * The video ref is also stored in the global viewer store for access by other components.
 * @param {string} src - The video source URL.
 * @returns {THREE.VideoTexture} - The video texture.
 *
 * Note:
 * This custom hook fixes an issue with useVideoTexture from `@react-three/drei`
 * where the video element is not properly cleaned up when the source changes,
 * leading to multiple video elements being created and potential memory leaks.
 * 
 * Unfortunately, I couldn't figure out how to extend or modify useVideoTexture directly, 
 * so this is a workaround that manually manages the video element and texture lifecycle. 
 * 
 * Their bug is reproduced here: https://codesandbox.io/p/github/mohamedtahaguelzim/drei-video-bug/master
 * I have not yet created an issue on their repo. Might do it later, or you can if you want.
 *
 * This hook supports HLS sources (.m3u8) as well as regular video sources.
 
 */
export const useCustomVideoTexture = (src) => {
  const setVideoRef = useViewerStore((s) => s.setVideoRef);
  const clearVideoRef = useViewerStore((s) => s.clearVideoRef);
  const [texture, setTexture] = useState(null);

  useEffect(() => {
    if (!src) return; // TODO: validate src format to prevent undesired behavior

    const video = document.createElement("video");
    video.crossOrigin = "anonymous";
    video.preload = "auto";
    video.playsInline = true;

    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    videoTexture.generateMipmaps = false;
    setTexture(videoTexture);

    setVideoRef(video);

    let hls;
    const isHls = typeof src === "string" && /\.m3u8($|\?)/i.test(src); // checks for .m3u8 at end or before query params
    const tryPlay = () => {
      const p = video.play();
      if (p && typeof p.then === "function") p.catch(() => {});
    };

    if (isHls) {
      if (isSupported()) {
        hls = new Hls({
          enableWorker: true,
          lowLatencyMode: true,
          backBufferLength: 90,
        });
        hls.attachMedia(video);
        hls.on(Hls.Events.MEDIA_ATTACHED, () => {
          hls.loadSource(src);
        });
        hls.on(Hls.Events.MANIFEST_PARSED, tryPlay);
      } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
        // TODO: check if this code is valid/reachable
        video.src = src; // Safari
        video.addEventListener("loadedmetadata", tryPlay, { once: true });
      } else {
        console.warn("HLS not supported in this browser.");
      }
    } else {
      video.src = src;
      video.addEventListener("loadedmetadata", tryPlay, { once: true });
    }

    return () => {
      // Clean up HLS instance
      if (hls) {
        try {
          hls.destroy();
        } catch (error) {
          console.warn("Error destroying HLS instance:", error);
        }
      }
      // Clean up video element
      if (video) {
        try {
          video.pause();
        } catch (error) {
          console.warn("Error pausing video:", error);
        }
        video.removeAttribute("src");
        try {
          video.load();
        } catch (error) {
          console.warn("Error unmounting video:", error);
        }
      }
      clearVideoRef();
      videoTexture.dispose();
    };
  }, [src, setVideoRef, clearVideoRef]);

  return texture;
};
