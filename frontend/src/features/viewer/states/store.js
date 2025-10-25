import { create } from "zustand";

/**
 * Player store using Zustand.
 * It holds the selected match, playback state, and video element reference.
 */
export const usePlayerStore = create((set) => ({
  // selected match object: { id, label, src, params, uniforms }
  selectedMatch: null,
  setSelectedMatch: (match) => set({ selectedMatch: match }),

  // playback state (kept minimal for now)
  playing: false,
  setPlaying: (p) => set({ playing: p }),

  // single video element ref (each match has at most one video)
  videoRef: null,
  setVideoRef: (el) => set({ videoRef: el }),
  clearVideoRef: () => set({ videoRef: null }),
}));

export default usePlayerStore;
