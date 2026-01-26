import { create } from 'zustand';
import { DEFAULT_COLOR_CORRECTION } from '../utils/utils';

/**
 * Player store using Zustand.
 * It holds the selected match, playback state, and video element reference.
 */
export const useViewerStore = create((set, get) => ({
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

	// fullscreen state
	fullscreen: false,
	setFullscreen: (fs) => set({ fullscreen: fs }),

	// Color correction state
	leftColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
	rightColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
	setLeftColorCorrection: (cc) => set({ leftColorCorrection: cc }),
	setRightColorCorrection: (cc) => set({ rightColorCorrection: cc }),
	resetColorCorrection: () =>
		set({
			leftColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
			rightColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
		}),

	// Load color correction from match metadata
	loadColorCorrectionFromMatch: () => {
		const match = get().selectedMatch;
		if (match?.metadata?.colorCorrection) {
			const { left, right } = match.metadata.colorCorrection;
			set({
				leftColorCorrection: left ? { ...DEFAULT_COLOR_CORRECTION, ...left } : { ...DEFAULT_COLOR_CORRECTION },
				rightColorCorrection: right
					? { ...DEFAULT_COLOR_CORRECTION, ...right }
					: { ...DEFAULT_COLOR_CORRECTION },
			});
		} else {
			// Reset to defaults if no saved color correction
			set({
				leftColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
				rightColorCorrection: { ...DEFAULT_COLOR_CORRECTION },
			});
		}
	},
}));

export default useViewerStore;
